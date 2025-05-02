import os
import numpy as np
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches
from pathlib import Path
import cv2
import coremltools as ct
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MobileSAMCoreMLTest')

class InteractiveMobileSAMCoreMLTest:
    def __init__(self, encoder_path, mask_decoder_path):
        self.encoder_path = encoder_path
        self.mask_decoder_path = mask_decoder_path
        self.current_image_idx = 0
        self.images = []
        self.current_points = []
        self.current_labels = []
        self.masks = []
        self.scores = []
        self.encoder_model = None
        self.mask_decoder_model = None
        self.current_embedding = None
        self.fig = None
        self.ax_image = None
        self.ax_mask = None
        self.ax_overlay = None
        self.image_plot = None
        self.mask_plot = None
        self.overlay_plot = None
        self.mode = "positive"  # Start with positive points
        
        # Store model input names for easier access
        self.embedding_input_name = "image"
        self.mask_decoder_input_names = ["image_embedding", "point_coords", "point_labels"]
        
        # Configuration
        self.results_dir = "coreml_interactive_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize plot
        self.init_plot()
        
    def load_models(self):
        """Load the MobileSAM CoreML models"""
        logger.info("Loading MobileSAM CoreML models...")
        
        try:
            # Load the embedding model
            self.encoder_model = ct.models.MLModel(self.encoder_path)
            logger.info(f"✅ Embedding model loaded from {self.encoder_path}")
            
            # Load the mask decoder model
            self.mask_decoder_model = ct.models.MLModel(self.mask_decoder_path)
            logger.info(f"✅ Mask decoder model loaded from {self.mask_decoder_path}")
            
            # Try to inspect model specifications
            try:
                logger.info("\nModel Specifications:")
                
                # Try to extract input/output names (handling different CoreML versions)
                try:
                    if hasattr(self.encoder_model, 'get_spec'):
                        embedding_inputs = [input.name for input in self.encoder_model.get_spec().description.input]
                        embedding_outputs = [output.name for output in self.encoder_model.get_spec().description.output]
                        
                        if embedding_inputs:
                            self.embedding_input_name = embedding_inputs[0]
                        
                        logger.info(f"Embedding Model Inputs: {embedding_inputs}")
                        logger.info(f"Embedding Model Outputs: {embedding_outputs}")
                    else:
                        logger.info("Could not access embedding model specifications via get_spec()")
                    
                    if hasattr(self.mask_decoder_model, 'get_spec'):
                        decoder_inputs = [input.name for input in self.mask_decoder_model.get_spec().description.input]
                        decoder_outputs = [output.name for output in self.mask_decoder_model.get_spec().description.output]
                        
                        if decoder_inputs and len(decoder_inputs) >= 3:
                            self.mask_decoder_input_names = decoder_inputs[:3]
                            
                        logger.info(f"Mask Decoder Model Inputs: {decoder_inputs}")
                        logger.info(f"Mask Decoder Model Outputs: {decoder_outputs}")
                        
                        # Log input shapes if available
                        try:
                            for input_spec in self.mask_decoder_model.get_spec().description.input:
                                input_name = input_spec.name
                                if hasattr(input_spec.type, 'multiArrayType'):
                                    input_shape = input_spec.type.multiArrayType.shape
                                    logger.info(f"Mask Decoder Input '{input_name}' shape: {input_shape}")
                        except Exception as e:
                            logger.warning(f"Could not access input shapes: {e}")
                    else:
                        logger.info("Could not access mask decoder model specifications via get_spec()")
                except Exception as e:
                    logger.warning(f"Error getting model specs via get_spec(): {e}")
                
                # Alternative method to get input/output info
                try:
                    if hasattr(self.encoder_model, 'input_description') and hasattr(self.encoder_model.input_description, 'keys'):
                        logger.info(f"Embedding Model Input Description Keys: {list(self.encoder_model.input_description.keys())}")
                    
                    if hasattr(self.mask_decoder_model, 'input_description') and hasattr(self.mask_decoder_model.input_description, 'keys'):
                        logger.info(f"Mask Decoder Model Input Description Keys: {list(self.mask_decoder_model.input_description.keys())}")
                except Exception as e:
                    logger.warning(f"Error getting model specs via input_description: {e}")
                
                # Log the input names we'll use
                logger.info(f"Will use embedding input name: {self.embedding_input_name}")
                logger.info(f"Will use mask decoder input names: {self.mask_decoder_input_names}")
                
            except Exception as e:
                logger.warning(f"Could not print full model specifications: {e}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            return False
            
    def load_images(self, image_paths=None):
        """Load images for testing"""
        self.images = []
        
        # If specific images are provided
        if image_paths and len(image_paths) > 0:
            for path in image_paths:
                try:
                    img = Image.open(path)
                    img = img.convert("RGB")
                    
                    # Resize to 1024x1024 (standard size for MobileSAM)
                    if img.width != 1024 or img.height != 1024:
                        img = img.resize((1024, 1024), Image.BILINEAR)
                    
                    # Store the PIL image and numpy array separately
                    img_np = np.array(img)
                    
                    img_info = {
                        "path": path,
                        "name": os.path.basename(path),
                        "image": img_np,
                        "pil_image": img,
                        "size": img.size
                    }
                    self.images.append(img_info)
                    logger.info(f"Loaded image: {path}")
                except Exception as e:
                    logger.error(f"Error loading image {path}: {e}")
        
        # If no images are loaded, create synthetic ones
        if len(self.images) == 0:
            logger.info("No images loaded, creating synthetic test images...")
            
            # Create synthetic test images
            shapes_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
            cv2.circle(shapes_image, (512, 512), 200, (0, 0, 255), -1)
            cv2.rectangle(shapes_image, (100, 100), (400, 400), (0, 255, 0), -1)
            pts = np.array([[700, 700], [900, 700], [800, 500]], np.int32)
            cv2.fillPoly(shapes_image, [pts], (255, 0, 0))
            
            shapes_path = os.path.join(self.results_dir, "synthetic_shapes.png")
            cv2.imwrite(shapes_path, shapes_image)
            
            # Convert to PIL Image
            shapes_pil = Image.fromarray(shapes_image)
            
            img_info = {
                "path": shapes_path,
                "name": "synthetic_shapes.png",
                "image": shapes_image,
                "pil_image": shapes_pil,
                "size": (1024, 1024)
            }
            self.images.append(img_info)
            logger.info(f"Created synthetic image: {shapes_path}")
            
            # Create another test image with gradients
            gradient_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
            for i in range(1024):
                gradient_image[:, i, 0] = i // 4  # Red channel
                gradient_image[i, :, 1] = i // 4  # Green channel
                gradient_image[i, i, 2] = 255     # Blue diagonal
            
            gradient_path = os.path.join(self.results_dir, "synthetic_gradient.png")
            cv2.imwrite(gradient_path, gradient_image)
            
            # Convert to PIL Image
            gradient_pil = Image.fromarray(gradient_image)
            
            img_info = {
                "path": gradient_path,
                "name": "synthetic_gradient.png",
                "image": gradient_image,
                "pil_image": gradient_pil,
                "size": (1024, 1024)
            }
            self.images.append(img_info)
            logger.info(f"Created synthetic image: {gradient_path}")
        
        return len(self.images) > 0
    
    def init_plot(self):
        """Initialize the matplotlib plot for interaction"""
        self.fig = plt.figure(figsize=(15, 6))
        self.fig.canvas.manager.set_window_title("Interactive MobileSAM CoreML Test")
        
        # Create axes for the images
        self.ax_image = plt.subplot(1, 3, 1)
        self.ax_image.set_title("Image (Click to add points)")
        self.ax_image.set_xticks([])
        self.ax_image.set_yticks([])
        
        self.ax_mask = plt.subplot(1, 3, 2)
        self.ax_mask.set_title("Mask")
        self.ax_mask.set_xticks([])
        self.ax_mask.set_yticks([])
        
        self.ax_overlay = plt.subplot(1, 3, 3)
        self.ax_overlay.set_title("Overlay")
        self.ax_overlay.set_xticks([])
        self.ax_overlay.set_yticks([])
        
        # Add controls
        plt.subplots_adjust(bottom=0.2)
        
        # Button positions
        button_width = 0.1
        button_height = 0.05
        button_spacing = 0.02
        
        # Clear button
        ax_clear = plt.axes([0.1, 0.05, button_width, button_height])
        self.clear_button = Button(ax_clear, 'Clear Points')
        self.clear_button.on_clicked(self.clear_points)
        
        # Toggle mode button
        ax_toggle = plt.axes([0.22, 0.05, button_width, button_height])
        self.toggle_button = Button(ax_toggle, 'Toggle +/-')
        self.toggle_button.on_clicked(self.toggle_mode)
        
        # Predict button
        ax_predict = plt.axes([0.34, 0.05, button_width, button_height])
        self.predict_button = Button(ax_predict, 'Predict')
        self.predict_button.on_clicked(self.predict_mask)
        
        # Save button
        ax_save = plt.axes([0.46, 0.05, button_width, button_height])
        self.save_button = Button(ax_save, 'Save Result')
        self.save_button.on_clicked(self.save_result)
        
        # Next image button
        ax_next = plt.axes([0.58, 0.05, button_width, button_height])
        self.next_button = Button(ax_next, 'Next Image')
        self.next_button.on_clicked(self.next_image)
        
        # Previous image button
        ax_prev = plt.axes([0.7, 0.05, button_width, button_height])
        self.prev_button = Button(ax_prev, 'Prev Image')
        self.prev_button.on_clicked(self.prev_image)
        
        # Recompute embedding button
        ax_recompute = plt.axes([0.82, 0.05, button_width, button_height])
        self.recompute_button = Button(ax_recompute, 'Recompute')
        self.recompute_button.on_clicked(self.recompute_embedding)
        
        # Add status text
        self.ax_status = plt.figtext(0.5, 0.12, "Ready", ha="center")
        self.mode_text = plt.figtext(0.5, 0.17, "Click Mode: Positive Points", ha="center")
        
        # Connect the click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        """Handle mouse clicks on the image"""
        if event.inaxes == self.ax_image:
            x, y = int(event.xdata), int(event.ydata)
            
            # Add point to the list
            self.current_points.append([x, y])
            
            # Add label (1 for positive, 0 for negative)
            label = 1 if self.mode == "positive" else 0
            self.current_labels.append(label)
            
            # Update the plot with the new point
            color = 'green' if label == 1 else 'red'
            self.ax_image.scatter(x, y, color=color, s=100, edgecolor='white')
            
            # Refresh the plot
            self.fig.canvas.draw_idle()
            
            # Show the coordinates
            self.ax_status.set_text(f"Added {'positive' if label == 1 else 'negative'} point at ({x}, {y})")
    
    def clear_points(self, event):
        """Clear all points"""
        self.current_points = []
        self.current_labels = []
        
        # Redraw the current image
        if len(self.images) > 0:
            self.display_image(self.current_image_idx)
        
        self.ax_status.set_text("Points cleared")
        self.fig.canvas.draw_idle()
    
    def toggle_mode(self, event):
        """Toggle between positive and negative point mode"""
        self.mode = "negative" if self.mode == "positive" else "positive"
        self.mode_text.set_text(f"Click Mode: {'Positive' if self.mode == 'positive' else 'Negative'} Points")
        self.ax_status.set_text(f"Mode changed to {'positive' if self.mode == 'positive' else 'negative'} points")
        self.fig.canvas.draw_idle()
    
    def compute_embedding(self, img_data):
        """Compute image embedding using the encoder model"""
        try:
            start_time = time.time()
            
            # Use the PIL image directly as required by the model
            pil_img = img_data["pil_image"]
            
            # Run the model with PIL image input
            logger.info(f"Running embedding model with input name: {self.embedding_input_name}")
            embedding_output = self.encoder_model.predict({self.embedding_input_name: pil_img})
            
            # Get the embedding from the output (first item)
            embedding_name = list(embedding_output.keys())[0]
            embedding = embedding_output[embedding_name]
            
            compute_time = time.time() - start_time
            self.ax_status.set_text(f"Embedding computed in {compute_time:.3f}s")
            
            logger.info(f"Embedding shape: {embedding.shape}, computed in {compute_time:.3f}s")
            return embedding
        except Exception as e:
            self.ax_status.set_text(f"Error computing embedding: {e}")
            logger.error(f"Error computing embedding: {e}", exc_info=True)
            return None
    
    def recompute_embedding(self, event):
        """Recompute the image embedding"""
        if len(self.images) == 0:
            self.ax_status.set_text("No image available")
            self.fig.canvas.draw_idle()
            return
        
        self.ax_status.set_text("Recomputing embedding...")
        self.fig.canvas.draw_idle()
        
        # Get current image data
        img_data = self.images[self.current_image_idx]
        
        # Compute embedding
        self.current_embedding = self.compute_embedding(img_data)
        
        if self.current_embedding is not None:
            self.ax_status.set_text(f"Embedding recomputed. Shape: {self.current_embedding.shape}")
        
        self.fig.canvas.draw_idle()
    
    def predict_mask(self, event):
        """Generate mask using the current points"""
        if len(self.images) == 0 or len(self.current_points) == 0:
            self.ax_status.set_text("No image or points available")
            self.fig.canvas.draw_idle()
            return
        
        if self.current_embedding is None:
            self.ax_status.set_text("No embedding available. Computing...")
            self.fig.canvas.draw_idle()
            
            # Compute embedding first
            img_data = self.images[self.current_image_idx]
            self.current_embedding = self.compute_embedding(img_data)
            
            if self.current_embedding is None:
                self.ax_status.set_text("Failed to compute embedding")
                self.fig.canvas.draw_idle()
                return
        
        self.ax_status.set_text("Predicting mask...")
        self.fig.canvas.draw_idle()
        
        # Try different formats for point coordinates and labels
        try:
            # Get current points and labels
            points = np.array(self.current_points)
            labels = np.array(self.current_labels)
            
            # Format the points based on the error message (need rank 2 for point_labels)
            # Try various formats
            formats_to_try = [
                # Format 1: Standard format from CoreML conversion script (rank 2)
                {
                    "point_coords": np.array([points], dtype=np.float32),  # [1, n_points, 2]
                    "point_labels": np.array([labels], dtype=np.float32)   # [1, n_points]
                },
                # Format 2: Using rank 2 for labels without extra batch dim
                {
                    "point_coords": np.array([points], dtype=np.float32),  # [1, n_points, 2]
                    "point_labels": labels.reshape(1, -1).astype(np.float32)  # [1, n_points]
                },
                # Format 3: Using rank 2 for all inputs
                {
                    "point_coords": points.astype(np.float32),  # [n_points, 2]
                    "point_labels": labels.reshape(-1, 1).astype(np.float32)  # [n_points, 1]
                },
                # Format 4: Using rank 3 for coords but rank 2 for labels
                {
                    "point_coords": np.array([points], dtype=np.float32),  # [1, n_points, 2]
                    "point_labels": labels.reshape(1, -1).astype(np.float32)  # [1, n_points]
                }
            ]
            
            # Try each format
            mask_output = None
            error_messages = []
            
            for i, format_data in enumerate(formats_to_try):
                try:
                    logger.info(f"Trying point format {i+1}:")
                    logger.info(f"  point_coords shape: {format_data['point_coords'].shape}")
                    logger.info(f"  point_labels shape: {format_data['point_labels'].shape}")
                    
                    # Prepare the full input dictionary
                    mask_inputs = {
                        self.mask_decoder_input_names[0]: self.current_embedding,  # image_embedding
                        self.mask_decoder_input_names[1]: format_data["point_coords"],  # point_coords
                        self.mask_decoder_input_names[2]: format_data["point_labels"]   # point_labels
                    }
                    
                    # Run prediction
                    mask_output = self.mask_decoder_model.predict(mask_inputs)
                    
                    # If we get here, the format worked
                    logger.info(f"Format {i+1} worked!")
                    break
                    
                except Exception as e:
                    error_message = str(e)
                    logger.warning(f"Format {i+1} failed: {error_message}")
                    error_messages.append(f"Format {i+1} error: {error_message}")
            
            # If all formats failed, raise an error
            if mask_output is None:
                raise Exception(f"All point formats failed. Errors: {'; '.join(error_messages)}")
            
            # Process the output
            mask_name = list(mask_output.keys())[0]
            masks = mask_output[mask_name]
            
            # Try to get scores if available (second output)
            scores = None
            if len(mask_output) > 1:
                score_name = list(mask_output.keys())[1]
                scores = mask_output[score_name]
            else:
                # If no scores in output, use default values
                scores = np.array([0.9])
            
            # Save the results
            self.masks = masks
            self.scores = scores
            
            # Process the mask based on its shape
            if len(masks.shape) == 4:  # [batch, num_masks, height, width]
                best_mask_idx = 0
                mask = masks[0, best_mask_idx]
                score = scores[best_mask_idx] if scores is not None and len(scores) > best_mask_idx else 0.9
            elif len(masks.shape) == 3:  # [batch, height, width]
                mask = masks[0]
                score = scores[0] if scores is not None and len(scores) > 0 else 0.9
            else:
                raise ValueError(f"Unexpected mask shape: {masks.shape}")
            
            logger.info(f"Generated mask with shape {masks.shape}, score {score:.3f}")
            
            # Display the mask
            if self.mask_plot is not None:
                self.mask_plot.set_data(mask)
            else:
                self.mask_plot = self.ax_mask.imshow(mask, cmap='gray')
            
            self.ax_mask.set_title(f"Mask (Score: {score:.3f})")
            
            # Create overlay
            current_img = self.images[self.current_image_idx]["image"]
            
            # Ensure mask has same dimensions as image
            if mask.shape != current_img.shape[:2]:
                mask_resized = cv2.resize(mask.astype(np.uint8), 
                                         (current_img.shape[1], current_img.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            
            # Create overlay
            overlay = current_img.copy()
            red_mask = np.zeros_like(current_img)
            red_mask[:, :, 0] = mask_resized * 255  # Red channel
            
            # Apply alpha blending
            alpha = 0.5
            cv2.addWeighted(red_mask, alpha, overlay, 1 - alpha, 0, overlay)
            
            # Display overlay
            if self.overlay_plot is not None:
                self.overlay_plot.set_data(overlay)
            else:
                self.overlay_plot = self.ax_overlay.imshow(overlay)
            
            # Update status
            self.ax_status.set_text(f"Prediction complete!")
            
            # Draw points on the overlay
            for i, (x, y) in enumerate(self.current_points):
                color = 'green' if self.current_labels[i] == 1 else 'red'
                self.ax_overlay.scatter(x, y, color=color, s=100, edgecolor='white')
            
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            self.ax_status.set_text(f"Error during prediction: {e}")
            logger.error(f"Error during prediction: {e}", exc_info=True)
            self.fig.canvas.draw_idle()
    
    def save_result(self, event):
        """Save the current result to disk"""
        if len(self.images) == 0 or not hasattr(self, 'masks') or len(self.masks) == 0:
            self.ax_status.set_text("No results to save")
            self.fig.canvas.draw_idle()
            return
        
        # Get current image name (without extension)
        img_name = os.path.splitext(self.images[self.current_image_idx]["name"])[0]
        
        # Create timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save the figure
        result_filename = f"{img_name}_result_{timestamp}.png"
        result_path = os.path.join(self.results_dir, result_filename)
        self.fig.savefig(result_path)
        
        # Save the masks
        if len(self.masks.shape) == 4:  # [batch, num_masks, height, width]
            for i in range(self.masks.shape[1]):
                mask_filename = f"{img_name}_mask_{i}_{timestamp}.png"
                mask_path = os.path.join(self.results_dir, mask_filename)
                plt.imsave(mask_path, self.masks[0, i], cmap='gray')
        else:  # [batch, height, width]
            mask_filename = f"{img_name}_mask_{timestamp}.png"
            mask_path = os.path.join(self.results_dir, mask_filename)
            plt.imsave(mask_path, self.masks[0], cmap='gray')
        
        # Save the points
        points_data = {
            "image": self.images[self.current_image_idx]["path"],
            "points": self.current_points,
            "labels": self.current_labels,
            "scores": self.scores.tolist() if hasattr(self.scores, 'tolist') else self.scores
        }
        
        # Save as text file
        points_filename = f"{img_name}_points_{timestamp}.txt"
        points_path = os.path.join(self.results_dir, points_filename)
        with open(points_path, 'w') as f:
            f.write(f"Image: {points_data['image']}\n")
            f.write("Points (x, y):\n")
            for i, (x, y) in enumerate(points_data['points']):
                label_type = "Positive" if points_data['labels'][i] == 1 else "Negative"
                f.write(f"{i+1}. ({x}, {y}) - {label_type}\n")
            f.write("\nScores:\n")
            for i, score in enumerate(points_data['scores']):
                f.write(f"Mask {i+1}: {score}\n")
        
        self.ax_status.set_text(f"Saved results to {self.results_dir}")
        self.fig.canvas.draw_idle()
    
    def next_image(self, event):
        """Move to the next image"""
        if len(self.images) > 0:
            self.current_image_idx = (self.current_image_idx + 1) % len(self.images)
            self.display_image(self.current_image_idx)
            self.current_points = []
            self.current_labels = []
            self.masks = []
            self.scores = []
            self.current_embedding = None
    
    def prev_image(self, event):
        """Move to the previous image"""
        if len(self.images) > 0:
            self.current_image_idx = (self.current_image_idx - 1) % len(self.images)
            self.display_image(self.current_image_idx)
            self.current_points = []
            self.current_labels = []
            self.masks = []
            self.scores = []
            self.current_embedding = None
    
    def display_image(self, idx):
        """Display the image at the given index"""
        if idx < 0 or idx >= len(self.images):
            return
        
        # Get the image
        img_data = self.images[idx]
        img = img_data["image"]
        
        # Update the image plot
        if self.image_plot is not None:
            self.image_plot.set_data(img)
        else:
            self.image_plot = self.ax_image.imshow(img)
        
        # Clear other plots
        if self.mask_plot is not None:
            self.mask_plot.set_data(np.zeros_like(img[:, :, 0]))
        
        if self.overlay_plot is not None:
            self.overlay_plot.set_data(np.zeros_like(img))
        
        # Update title
        self.ax_image.set_title(f"Image: {img_data['name']} (Click to add points)")
        self.ax_mask.set_title("Mask")
        self.ax_overlay.set_title("Overlay")
        
        # Compute embedding for this image
        self.ax_status.set_text(f"Computing embedding for {img_data['name']}...")
        self.fig.canvas.draw_idle()
        
        self.current_embedding = self.compute_embedding(img_data)
        
        if self.current_embedding is not None:
            self.ax_status.set_text(f"Loaded image: {img_data['name']} (embedding shape: {self.current_embedding.shape})")
        else:
            self.ax_status.set_text(f"Loaded image: {img_data['name']} (embedding failed)")
        
        # Refresh the plot
        self.fig.canvas.draw_idle()
    
    def run(self, image_paths=None):
        """Run the interactive test"""
        # Load the models
        if not self.load_models():
            logger.error("Failed to load models, exiting.")
            return False
        
        # Load images
        if not self.load_images(image_paths):
            logger.error("Failed to load any images, exiting.")
            return False
        
        # Display the first image
        if len(self.images) > 0:
            self.display_image(0)
        
        # Show the plot
        plt.show()
        
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive MobileSAM CoreML Test")
    parser.add_argument("--encoder", type=str, default="coreml_models/MobileSAMEmbedding.mlpackage", 
                      help="Path to embedding model (.mlpackage)")
    parser.add_argument("--mask-decoder", type=str, default="coreml_models/MobileSAMMaskDecoder.mlpackage", 
                      help="Path to mask decoder model (.mlpackage)")
    parser.add_argument("--images", type=str, nargs='+', default=None,
                      help="Paths to test images (can provide multiple)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Run the interactive test
    test = InteractiveMobileSAMCoreMLTest(args.encoder, args.mask_decoder)
    test.run(args.images)