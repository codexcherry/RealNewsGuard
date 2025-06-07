from PIL import Image
import os
import numpy as np
import hashlib
from PIL.ExifTags import TAGS
from io import BytesIO
import requests

def process_image(image_path):
    """
    Process and analyze an image for potential manipulation.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: The image path (may be modified if image is processed)
    """
    if not image_path or not os.path.exists(image_path):
        return None
    
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Basic image info
        image_info = {
            "format": img.format,
            "size": img.size,
            "mode": img.mode,
        }
        
        # Extract EXIF data if available
        exif_data = extract_exif_data(img)
        
        # Check for metadata inconsistencies
        metadata_issues = check_metadata_issues(exif_data)
        
        # Check for error level analysis (ELA) indicators
        ela_score = error_level_analysis(img, image_path)
        
        # Store analysis results alongside the image
        analysis_path = image_path + ".analysis.txt"
        with open(analysis_path, "w") as f:
            f.write(f"Image Info: {str(image_info)}\n")
            f.write(f"EXIF Data: {str(exif_data)}\n")
            f.write(f"Metadata Issues: {str(metadata_issues)}\n")
            f.write(f"ELA Score: {ela_score}\n")
        
        return image_path
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return image_path

def extract_exif_data(img):
    """
    Extract EXIF metadata from an image.
    
    Args:
        img (PIL.Image): Image object
        
    Returns:
        dict: EXIF metadata
    """
    exif_data = {}
    try:
        if hasattr(img, '_getexif') and img._getexif():
            for tag, value in img._getexif().items():
                tag_name = TAGS.get(tag, tag)
                exif_data[tag_name] = value
    except Exception as e:
        print(f"Error extracting EXIF data: {str(e)}")
    
    return exif_data

def check_metadata_issues(exif_data):
    """
    Check for inconsistencies or suspicious patterns in image metadata.
    
    Args:
        exif_data (dict): EXIF metadata
        
    Returns:
        dict: Issues found in metadata
    """
    issues = {}
    
    # Check for missing essential metadata
    essential_tags = ["DateTimeOriginal", "Make", "Model"]
    for tag in essential_tags:
        if tag not in exif_data:
            issues[f"missing_{tag.lower()}"] = True
    
    # Check for software used to edit the image
    if "Software" in exif_data:
        editing_software = ["photoshop", "gimp", "lightroom", "affinity", "pixlr"]
        for software in editing_software:
            if software.lower() in str(exif_data["Software"]).lower():
                issues["edited_with_software"] = exif_data["Software"]
    
    # Check for modification date vs. original date
    if "DateTimeOriginal" in exif_data and "DateTime" in exif_data:
        if exif_data["DateTimeOriginal"] != exif_data["DateTime"]:
            issues["date_modified"] = {
                "original": exif_data["DateTimeOriginal"],
                "modified": exif_data["DateTime"]
            }
    
    return issues

def error_level_analysis(img, image_path, quality=90):
    """
    Perform Error Level Analysis (ELA) to detect image manipulation.
    
    Args:
        img (PIL.Image): Original image
        image_path (str): Path to the image
        quality (int): JPEG quality for ELA
        
    Returns:
        float: ELA score (higher values indicate potential manipulation)
    """
    try:
        # Save the image with specified quality
        temp_path = image_path + ".temp.jpg"
        img.save(temp_path, "JPEG", quality=quality)
        
        # Open the saved image
        saved_img = Image.open(temp_path)
        
        # Convert images to numpy arrays
        original_array = np.array(img.convert("RGB")).astype(float)
        saved_array = np.array(saved_img.convert("RGB")).astype(float)
        
        # Calculate absolute difference
        diff = np.abs(original_array - saved_array)
        
        # Calculate ELA score (mean difference across all pixels and channels)
        ela_score = np.mean(diff)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return float(ela_score)
    
    except Exception as e:
        print(f"Error performing ELA: {str(e)}")
        return 0.0

def detect_image_source(image_path):
    """
    Attempt to detect the source of an image using reverse image search.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        dict: Image source information
    """
    # This is a placeholder function
    # In a real implementation, you would use a reverse image search API
    # such as Google Images, TinEye, or a custom solution
    
    return {
        "found": False,
        "message": "Image source detection requires external API integration"
    }

def compute_image_hash(image_path):
    """
    Compute a perceptual hash of an image for comparison.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        str: Image hash
    """
    try:
        # Open the image and resize to 8x8
        img = Image.open(image_path).convert("L").resize((8, 8), Image.LANCZOS)
        
        # Get pixel data
        pixels = list(img.getdata())
        
        # Compute average pixel value
        avg_pixel = sum(pixels) / len(pixels)
        
        # Create binary hash (1 if pixel > avg, 0 otherwise)
        bits = "".join(['1' if pixel > avg_pixel else '0' for pixel in pixels])
        
        # Convert binary string to hexadecimal
        hex_hash = hex(int(bits, 2))[2:].zfill(16)
        
        return hex_hash
    
    except Exception as e:
        print(f"Error computing image hash: {str(e)}")
        return None

def download_image_from_url(url, save_path=None):
    """
    Download an image from a URL.
    
    Args:
        url (str): Image URL
        save_path (str): Path to save the image
        
    Returns:
        str: Path to the saved image
    """
    try:
        # Make request
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Create image from response content
        img = Image.open(BytesIO(response.content))
        
        # Generate save path if not provided
        if not save_path:
            # Create a filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()
            extension = url.split('.')[-1] if '.' in url else 'jpg'
            if extension.lower() not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
                extension = 'jpg'
            save_path = f"uploads/{url_hash}.{extension}"
        
        # Save the image
        img.save(save_path)
        
        return save_path
    
    except Exception as e:
        print(f"Error downloading image from URL: {str(e)}")
        return None 