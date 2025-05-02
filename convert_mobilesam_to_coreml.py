import torch
import coremltools as ct
from mobile_sam import sam_model_registry
from mobile_sam.build_sam import build_sam_vit_t
import os
import numpy as np

print("Starting MobileSAM conversion to Core ML (both Image Encoder and Prompt Encoder/Mask Decoder)...")

# Register MobileSAM model
sam_model_registry["mobile_sam"] = build_sam_vit_t

# Load MobileSAM
weights_path = os.path.join("mobile_sam_repo", "weights", "mobile_sam.pt")
mobile_sam = sam_model_registry["mobile_sam"](checkpoint=weights_path)
mobile_sam.eval()

# Print model structure to understand it better
print("=== MobileSAM Model Structure ===")
print(f"Main components: {[name for name, _ in mobile_sam.named_children()]}")
for name, module in mobile_sam.named_children():
    print(f"\n{name} structure: {[n for n, _ in module.named_children()]}")

# =======================================================================
# PART 1: Convert Image Encoder to CoreML
# =======================================================================
print("\nConverting Image Encoder...")

# Define input shape for image encoder
input_shape = (1, 3, 1024, 1024)  # (batch_size, channels, height, width)

# Create a dummy input tensor
dummy_input = torch.rand(*input_shape)

# Trace the image encoder with the dummy input
traced_image_encoder = torch.jit.trace(mobile_sam.image_encoder, dummy_input)

# Convert image encoder to CoreML
image_encoder_mlmodel = ct.convert(
    traced_image_encoder,
    inputs=[ct.TensorType(name="image", shape=input_shape)],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16
)

# Save the image encoder model
image_encoder_mlmodel.save("MobileSAM_ImageEncoder.mlpackage")
print("Image Encoder saved as MobileSAM_ImageEncoder.mlpackage")

# =======================================================================
# PART 2: Convert Prompt Encoder separately (if exists)
# =======================================================================
if hasattr(mobile_sam, 'prompt_encoder'):
    print("\nConverting Prompt Encoder...")
    
    # Create a sample point input
    point_coords = torch.tensor([[[500, 500]]], dtype=torch.float)  # [batch_size, num_points, 2]
    point_labels = torch.tensor([[1]], dtype=torch.float)  # [batch_size, num_points]
    
    # Create a wrapper for the prompt encoder
    class PromptEncoderWrapper(torch.nn.Module):
        def __init__(self, prompt_encoder):
            super().__init__()
            self.prompt_encoder = prompt_encoder
            
        def forward(self, point_coords, point_labels):
            return self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None
            )
    
    # Create and trace prompt encoder wrapper
    prompt_encoder_wrapper = PromptEncoderWrapper(mobile_sam.prompt_encoder)
    prompt_encoder_wrapper.eval()
    
    try:
        traced_prompt_encoder = torch.jit.trace(prompt_encoder_wrapper, (point_coords, point_labels))
        
        # Convert prompt encoder to CoreML
        prompt_encoder_mlmodel = ct.convert(
            traced_prompt_encoder,
            inputs=[
                ct.TensorType(name="point_coords", shape=point_coords.shape),
                ct.TensorType(name="point_labels", shape=point_labels.shape)
            ],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS16
        )
        
        # Save the prompt encoder model
        prompt_encoder_mlmodel.save("MobileSAM_PromptEncoder.mlpackage")
        print("Prompt Encoder saved as MobileSAM_PromptEncoder.mlpackage")
    except Exception as e:
        print(f"Error tracing prompt encoder: {e}")
        print("Will try to include prompt encoder with mask decoder")

# =======================================================================
# PART 3: Convert Mask Decoder to CoreML
# =======================================================================

print("\n=== Mask Decoder Structure ===")
print(f"Mask Decoder attributes: {dir(mobile_sam.mask_decoder)}")
print(f"Mask Decoder forward args: {mobile_sam.mask_decoder.forward.__code__.co_varnames}")

# Create a dummy input for the image encoder
dummy_input = torch.rand(1, 3, 1024, 1024)  # (batch_size, channels, height, width)

# Get image embedding
with torch.no_grad():
    image_embedding = mobile_sam.image_encoder(dummy_input)
    print(f"Image embedding shape: {image_embedding.shape}")

# Create sample point coords and labels
point_coords = torch.tensor([[[500, 500]]], dtype=torch.float)  # [batch_size, num_points, 2]
point_labels = torch.tensor([[1]], dtype=torch.float)  # [batch_size, num_points]

# Get prompt embeddings from the prompt encoder
with torch.no_grad():
    sparse_embeddings, dense_embeddings = mobile_sam.prompt_encoder(
        points=(point_coords, point_labels),
        boxes=None,
        masks=None
    )
    print(f"Sparse embeddings shape: {sparse_embeddings.shape}")
    print(f"Dense embeddings shape: {dense_embeddings.shape}")

# Create a wrapper for the mask decoder that takes the proper inputs
class MaskDecoderWrapper(torch.nn.Module):
    def __init__(self, mask_decoder, prompt_encoder):
        super().__init__()
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, image_embedding, sparse_embeddings, dense_embeddings):
        # This signature should match what mask_decoder.forward expects
        masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        return masks, iou_predictions

# Create the wrapper
mask_decoder_wrapper = MaskDecoderWrapper(mobile_sam.mask_decoder, mobile_sam.prompt_encoder)
mask_decoder_wrapper.eval()

# Trace the mask decoder
try:
    print("\nTracing mask decoder...")
    # Create inputs for tracing
    mask_decoder_inputs = (image_embedding, sparse_embeddings, dense_embeddings)
    
    # Trace the model
    traced_mask_decoder = torch.jit.trace(mask_decoder_wrapper, mask_decoder_inputs)
    
    # Convert to CoreML
    print("Converting mask decoder to CoreML...")
    mask_decoder_mlmodel = ct.convert(
        traced_mask_decoder,
        inputs=[
            ct.TensorType(name="image_embedding", shape=image_embedding.shape),
            ct.TensorType(name="sparse_embeddings", shape=sparse_embeddings.shape),
            ct.TensorType(name="dense_embeddings", shape=dense_embeddings.shape)
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16
    )
    
    # Save the model
    mask_decoder_mlmodel.save("MobileSAM_MaskDecoder.mlpackage")
    print("Mask Decoder successfully saved as MobileSAM_MaskDecoder.mlpackage")
    
except Exception as e:
    print(f"Error converting mask decoder: {e}")
    
    # Try a different approach - create a complete pipeline that includes the prompt encoder
    print("\nTrying alternative approach with combined prompt and mask decoder...")
    
    class CombinedDecoderWrapper(torch.nn.Module):
        def __init__(self, prompt_encoder, mask_decoder):
            super().__init__()
            self.prompt_encoder = prompt_encoder
            self.mask_decoder = mask_decoder
        
        def forward(self, image_embedding, point_coords, point_labels):
            # Process points through prompt encoder
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None
            )
            
            # Process embeddings through mask decoder
            masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            
            return masks, iou_predictions
    
    # Create the combined wrapper
    combined_wrapper = CombinedDecoderWrapper(
        prompt_encoder=mobile_sam.prompt_encoder,
        mask_decoder=mobile_sam.mask_decoder
    )
    combined_wrapper.eval()
    
    try:
        # Trace the combined model
        combined_inputs = (image_embedding, point_coords, point_labels)
        traced_combined = torch.jit.trace(combined_wrapper, combined_inputs)
        
        # Convert to CoreML
        combined_mlmodel = ct.convert(
            traced_combined,
            inputs=[
                ct.TensorType(name="image_embedding", shape=image_embedding.shape),
                ct.TensorType(name="point_coords", shape=point_coords.shape),
                ct.TensorType(name="point_labels", shape=point_labels.shape)
            ],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS16
        )
        
        # Save the model
        combined_mlmodel.save("MobileSAM_CombinedDecoder.mlpackage")
        print("Combined Decoder successfully saved as MobileSAM_CombinedDecoder.mlpackage")
        
    except Exception as e:
        print(f"Error with combined approach: {e}")
        print("\nPlease inspect the MobileSAM architecture further and adjust accordingly.")

print("\nConversion process completed.")