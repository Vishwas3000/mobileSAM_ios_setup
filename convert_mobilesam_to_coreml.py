import torch
import coremltools as ct
from mobile_sam import SamPredictor, sam_model_registry
from mobile_sam.build_sam import build_sam_vit_t
import os

print("Starting MobileSAM conversion to Core ML...")

# Register MobileSAM model
sam_model_registry["mobile_sam"] = build_sam_vit_t

# Load MobileSAM
weights_path = os.path.join("mobile_sam_repo", "weights", "mobile_sam.pt")
mobile_sam = sam_model_registry["mobile_sam"](checkpoint=weights_path)
mobile_sam.eval()

# Create sample inputs
image_embedding = torch.rand(1, 256, 64, 64)
point_coords = torch.randint(low=0, high=1024, size=(1, 1, 2))
point_labels = torch.randint(low=0, high=1, size=(1, 1))

# Define models for conversion
class MobileSAMEmbedding(torch.nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.sam_model = sam_model
        
    def forward(self, x):
        return self.sam_model.image_encoder(x)

class MobileSAMMaskDecoder(torch.nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.sam_model = sam_model
        
    def forward(self, image_embeddings, point_coords, point_labels):
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        
        masks, scores = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        return masks, scores

# Create models
embedding_model = MobileSAMEmbedding(mobile_sam)
mask_model = MobileSAMMaskDecoder(mobile_sam)

# Convert to TorchScript
print("Converting models to TorchScript...")
example_image = torch.rand(1, 3, 1024, 1024)
embedding_model_traced = torch.jit.trace(embedding_model, example_image)
mask_model_traced = torch.jit.trace(mask_model, (image_embedding, point_coords, point_labels))

# Convert embedding model
print("Converting embedding model...")
embedding_model_coreml = ct.convert(
    embedding_model_traced,
    source="pytorch",
    inputs=[ct.ImageType(name="image", shape=(1, 3, 1024, 1024))],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL
)

# Save embedding model
output_dir = "coreml_models"
os.makedirs(output_dir, exist_ok=True)
embedding_model_coreml.save(os.path.join(output_dir, "MobileSAMEmbedding.mlpackage"))

# Convert mask model
print("Converting mask decoder model...")
mask_model_coreml = ct.convert(
    mask_model_traced,
    source="pytorch",
    inputs=[
        ct.TensorType(name="image_embeddings", shape=image_embedding.shape),
        ct.TensorType(name="point_coords", shape=point_coords.shape),
        ct.TensorType(name="point_labels", shape=point_labels.shape)
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL
)

# Save mask model
mask_model_coreml.save(os.path.join(output_dir, "MobileSAMMaskDecoder.mlpackage"))

print("Conversion complete! Models saved to:", output_dir)