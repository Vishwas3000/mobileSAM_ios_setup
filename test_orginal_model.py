import os
import numpy as np
import torch
import time
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Import MobileSAM components
from mobile_sam import SamPredictor, sam_model_registry
from mobile_sam.build_sam import build_sam_vit_t

def test_mobilesam():
    """
    Comprehensive test for the MobileSAM model, evaluating performance
    and segmentation quality on different inputs.
    """
    print("="*80)
    print(" MOBILESAM MODEL TEST ".center(80, "="))
    print("="*80)
    
    # Create output directory for test results
    results_dir = "mobilesam_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load the MobileSAM model
    print("\n1. Loading MobileSAM model...")
    weights_path = os.path.join("mobile_sam_repo", "weights", "mobile_sam.pt")
    
    try:
        # Register model
        sam_model_registry["mobile_sam"] = build_sam_vit_t
        
        # Load the model
        mobile_sam = sam_model_registry["mobile_sam"](checkpoint=weights_path)
        mobile_sam.eval()
        
        # Create the predictor
        predictor = SamPredictor(mobile_sam)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # 2. Prepare test images
    print("\n2. Preparing test images...")
    test_images = []
    
    # 2.1 Add real images if available
    test_images_dir = Path("test_images")
    if test_images_dir.exists():
        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        for img_path in image_files[:5]:  # Test up to 5 images
            try:
                print(f"  Loading {img_path.name}...")
                img = Image.open(img_path)
                img = img.convert("RGB")
                img = img.resize((1024, 1024))
                img_np = np.array(img)
                
                test_images.append({
                    "name": img_path.name,
                    "image": img_np
                })
            except Exception as e:
                print(f"  ⚠️ Couldn't load {img_path.name}: {e}")
    
    # 2.2 Create synthetic test images if needed
    if len(test_images) < 2:
        print("  Creating synthetic test images...")
        
        # Create an image with basic shapes
        shapes_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        # Add a red circle
        cv2.circle(shapes_image, (512, 512), 200, (0, 0, 255), -1)
        # Add a green rectangle
        cv2.rectangle(shapes_image, (100, 100), (400, 400), (0, 255, 0), -1)
        # Add a blue triangle
        pts = np.array([[700, 700], [900, 700], [800, 500]], np.int32)
        cv2.fillPoly(shapes_image, [pts], (255, 0, 0))
        
        test_images.append({
            "name": "synthetic_shapes.png",
            "image": shapes_image
        })
        
        # Create a gradient image
        gradient_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        for i in range(1024):
            gradient_image[:, i, 0] = i // 4  # Red channel
            gradient_image[i, :, 1] = i // 4  # Green channel
            gradient_image[i, i, 2] = 255     # Blue diagonal
        
        test_images.append({
            "name": "synthetic_gradient.png",
            "image": gradient_image
        })
        
        # Save the synthetic images
        os.makedirs("test_images", exist_ok=True)
        cv2.imwrite("test_images/synthetic_shapes.png", shapes_image)
        cv2.imwrite("test_images/synthetic_gradient.png", gradient_image)
    
    # 3. Define test prompts
    print("\n3. Defining test prompts...")
    
    test_prompts = [
        {
            "name": "single_point",
            "points": np.array([[512, 512]]),
            "labels": np.array([1])
        },
        {
            "name": "two_points_positive",
            "points": np.array([[400, 370], [900, 370]]),
            "labels": np.array([1, 1])
        },
        {
            "name": "three_mixed_points",
            "points": np.array([[400, 370], [900, 370], [650, 500]]),
            "labels": np.array([1, 1, 0])
        },
        {
            "name": "box_prompt",
            "box": np.array([400, 400, 600, 600])  # x1, y1, x2, y2
        }
    ]
    
    # 4. Run tests
    print("\n4. Running tests...")
    
    test_results = []
    
    for img_idx, img_data in enumerate(test_images):
        img_name = img_data["name"]
        img = img_data["image"]
        
        print(f"\n  Testing image [{img_idx+1}/{len(test_images)}]: {img_name}")
        
        # Time the set_image call
        set_image_start = time.time()
        predictor.set_image(img)
        set_image_time = time.time() - set_image_start
        print(f"  set_image time: {set_image_time:.4f}s")
        
        # Test each prompt type
        for prompt_idx, prompt in enumerate(test_prompts):
            prompt_name = prompt["name"]
            
            print(f"    Testing prompt [{prompt_idx+1}/{len(test_prompts)}]: {prompt_name}")
            
            # Run prediction based on prompt type
            predict_start = time.time()
            
            if "box" in prompt:
                # Box prompt
                input_box = prompt["box"]
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],  # Add batch dimension
                    multimask_output=True
                )
            else:
                # Point prompts
                input_points = prompt["points"]
                input_labels = prompt["labels"]
                masks, scores, logits = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )
            
            predict_time = time.time() - predict_start
            print(f"    predict time: {predict_time:.4f}s")
            
            # Process results
            num_masks = len(masks)
            print(f"    generated {num_masks} masks")
            print(f"    scores: {scores}")
            
            # Record test results
            test_result = {
                "image": img_name,
                "prompt": prompt_name,
                "set_image_time": set_image_time,
                "predict_time": predict_time,
                "num_masks": num_masks,
                "scores": scores.tolist()
            }
            test_results.append(test_result)
            
            # Create visualization
            fig, axes = plt.subplots(1, num_masks + 1, figsize=(5 * (num_masks + 1), 5))
            
            # Original image with prompt
            axes[0].imshow(img)
            if "box" in prompt:
                # Draw box
                box = prompt["box"]
                rect = plt.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                axes[0].add_patch(rect)
            else:
                # Draw points
                for i, (x, y) in enumerate(prompt["points"]):
                    color = 'green' if prompt["labels"][i] == 1 else 'red'
                    axes[0].scatter(x, y, color=color, s=100, edgecolor='white')
            
            axes[0].set_title("Input Image with Prompt")
            axes[0].axis('off')
            
            # Show each mask with its score
            for mask_idx in range(num_masks):
                # Show the mask
                axes[mask_idx + 1].imshow(img)
                
                # Create a mask overlay
                show_mask = np.zeros_like(img, dtype=np.uint8)
                mask_array = masks[mask_idx]
                
                # Convert binary mask to RGB overlay
                color_mask = np.array([30, 144, 255], dtype=np.uint8)  # Dodger Blue
                show_mask[mask_array] = color_mask
                
                # Apply transparency
                mask_image = Image.fromarray(show_mask)
                img_pil = Image.fromarray(img)
                overlay = Image.blend(img_pil, mask_image, 0.5)
                
                axes[mask_idx + 1].imshow(np.array(overlay))
                axes[mask_idx + 1].set_title(f"Mask {mask_idx+1}\nScore: {scores[mask_idx]:.3f}")
                axes[mask_idx + 1].axis('off')
            
            plt.tight_layout()
            
            # Save the visualization
            viz_filename = f"{img_name.split('.')[0]}_{prompt_name}_results.png"
            plt.savefig(os.path.join(results_dir, viz_filename))
            plt.close()
            
            print(f"    Saved visualization to: {os.path.join(results_dir, viz_filename)}")
    
    # 5. Test performance with varied number of points
    print("\n5. Testing performance with variable number of points...")
    variable_point_results = []
    
    # Use the first image for this test
    test_image = test_images[0]["image"]
    predictor.set_image(test_image)
    
    # Test with increasing number of points
    for num_points in [1, 2, 5, 10, 20]:
        print(f"  Testing with {num_points} points...")
        
        # Generate random points
        rng = np.random.RandomState(42)  # For reproducibility
        points = rng.randint(100, 900, size=(num_points, 2))
        labels = np.ones(num_points)  # All positive points
        
        # Time the prediction
        start_time = time.time()
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )
        predict_time = time.time() - start_time
        
        print(f"  Prediction time with {num_points} points: {predict_time:.4f}s")
        
        variable_point_results.append({
            "num_points": num_points,
            "predict_time": predict_time
        })
        
        # Create visualization
        plt.figure(figsize=(10, 5))
        
        # Left: Image with points
        plt.subplot(1, 2, 1)
        plt.imshow(test_image)
        for x, y in points:
            plt.scatter(x, y, color='green', s=50, edgecolor='white')
        plt.title(f"{num_points} Points")
        plt.axis('off')
        
        # Right: Segmentation result
        plt.subplot(1, 2, 2)
        plt.imshow(test_image)
        show_mask = np.zeros_like(test_image, dtype=np.uint8)
        color_mask = np.array([30, 144, 255], dtype=np.uint8)
        show_mask[masks[0]] = color_mask
        
        mask_image = Image.fromarray(show_mask)
        img_pil = Image.fromarray(test_image)
        overlay = Image.blend(img_pil, mask_image, 0.5)
        
        plt.imshow(np.array(overlay))
        plt.title(f"Mask (Score: {scores[0]:.3f})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"variable_points_{num_points}.png"))
        plt.close()
    
    # Create performance plot
    plt.figure(figsize=(10, 6))
    x = [r["num_points"] for r in variable_point_results]
    y = [r["predict_time"] for r in variable_point_results]
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2)
    plt.xlabel("Number of Points")
    plt.ylabel("Prediction Time (seconds)")
    plt.title("MobileSAM Performance vs Number of Points")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "performance_vs_points.png"))
    plt.close()
    
    # 6. Generate summary report
    print("\n6. Generating summary report...")
    
    # Calculate statistics
    avg_set_image_time = np.mean([r["set_image_time"] for r in test_results])
    avg_predict_time = np.mean([r["predict_time"] for r in test_results])
    
    # Print summary
    print("\n" + "="*50)
    print(" TEST SUMMARY ".center(50, "="))
    print("="*50)
    print(f"Images tested: {len(test_images)}")
    print(f"Prompt types tested: {len(test_prompts)}")
    print(f"Average set_image time: {avg_set_image_time:.4f}s")
    print(f"Average predict time: {avg_predict_time:.4f}s")
    
    # Save detailed report
    with open(os.path.join(results_dir, "mobilesam_test_report.txt"), "w") as f:
        f.write("MOBILESAM MODEL TEST REPORT\n")
        f.write("==========================\n\n")
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Images tested: {len(test_images)}\n")
        f.write(f"Prompt types tested: {len(test_prompts)}\n")
        f.write(f"Average set_image time: {avg_set_image_time:.4f}s\n")
        f.write(f"Average predict time: {avg_predict_time:.4f}s\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("----------------\n")
        
        for img_name in set(r["image"] for r in test_results):
            f.write(f"\nImage: {img_name}\n")
            img_results = [r for r in test_results if r["image"] == img_name]
            
            f.write(f"  set_image time: {img_results[0]['set_image_time']:.4f}s\n")
            
            for r in img_results:
                f.write(f"  Prompt: {r['prompt']}\n")
                f.write(f"    predict time: {r['predict_time']:.4f}s\n")
                f.write(f"    masks generated: {r['num_masks']}\n")
                f.write(f"    scores: {r['scores']}\n")
        
        f.write("\nVARIABLE POINT PERFORMANCE:\n")
        f.write("---------------------------\n")
        for r in variable_point_results:
            f.write(f"  {r['num_points']} points: {r['predict_time']:.4f}s\n")
    
    print(f"\nDetailed report saved to: {os.path.join(results_dir, 'mobilesam_test_report.txt')}")
    
    # 7. Create comparison grid for different prompt types
    try:
        # Use the first image to compare different prompt types
        first_image = test_images[0]["name"]
        prompt_results = [r for r in test_results if r["image"] == first_image]
        
        if prompt_results:
            plt.figure(figsize=(len(prompt_results) * 5, 5))
            
            for i, result in enumerate(prompt_results):
                plt.subplot(1, len(prompt_results), i + 1)
                
                # Load the result image
                result_filename = f"{first_image.split('.')[0]}_{result['prompt']}_results.png"
                result_path = os.path.join(results_dir, result_filename)
                
                if os.path.exists(result_path):
                    result_img = plt.imread(result_path)
                    plt.imshow(result_img)
                    plt.title(f"{result['prompt']}")
                    plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "prompt_comparison.png"))
            plt.close()
            
            print(f"Prompt comparison saved to: {os.path.join(results_dir, 'prompt_comparison.png')}")
    except Exception as e:
        print(f"Failed to create prompt comparison: {e}")
    
    print("\n✅ MobileSAM testing completed successfully!")
    return True

if __name__ == "__main__":
    test_mobilesam()