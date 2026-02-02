import os
import shutil
import sys

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available")

source = r"C:\Users\user\Downloads\Heartfelt\WhatsApp Image 2026-02-02 at 11.58.43 PM.jpeg"
dest_dir = r"C:\Users\user\Downloads\Heartfelt\app\static\images"
dest = os.path.join(dest_dir, "logo.jpeg")

# Ensure dir exists (redundant but safe)
os.makedirs(dest_dir, exist_ok=True)

# Copy file
shutil.copy2(source, dest)
print(f"Copied image to {dest}")

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

if PIL_AVAILABLE:
    try:
        img = Image.open(dest)
        img = img.resize((150, 150))
        result = img.convert('P', palette=Image.ADAPTIVE, colors=5)
        result.putalpha(0)
        colors = result.getcolors(150*150)
        # Sort by count
        colors.sort(key=lambda x: x[0], reverse=True)
        
        print("Dominant colors:")
        for count, col in colors[:3]:
            print(f"{col}: {rgb_to_hex(col[:3])}")
    except Exception as e:
        print(f"Error processing image: {e}")
