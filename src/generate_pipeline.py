import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patheffects import withStroke
from PIL import Image
import os

# ==============================================================================
# MODIFICATION: Point this to your local retinal image directory
# Use forward slashes (/) or double backslashes (\\) for Windows paths.
# Example: "C:/Users/Name/Desktop/retina_images/fundus_01.jpg"
# Or relative path: "./my_images/fundus.jpg"
# ==============================================================================
local_image_path = "path/to/your/retina_image.jpg" 

# Load the image locally
if os.path.exists(local_image_path):
    eye_img = Image.open(local_image_path)
else:
    raise FileNotFoundError(f"Could not find the image at: {local_image_path}. Please check the path.")

# 1. Setup the canvas
fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
ax.axis('off')
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)

# Define colors
c_legacy = '#E3F2FD' # Light Blue
c_update = '#FFEBEE' # Light Red
c_cure = '#E8F5E9'   # Light Green
c_output = '#F3E5F5' # Light Purple
c_text = '#333333'

# --- HELPER FUNCTIONS ---
def draw_box(ax, x, y, w, h, color, title, text_lines, border_color):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                  fc=color, ec=border_color, lw=2.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.4, title, ha='center', va='top', 
            fontsize=14, fontweight='bold', color=border_color, zorder=3)
    for i, line in enumerate(text_lines):
        ax.text(x + w/2, y + h - 1.2 - (i * 0.5), line, ha='center', va='top', 
                fontsize=11, color=c_text, zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color='#555555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=2.5, mutation_scale=20), zorder=1)

# --- TITLE ---
ax.text(9, 9.5, 'CURE: Regression-Free Updating of Retinal Disease Classifiers', 
        ha='center', va='center', fontsize=20, fontweight='bold', color='#000000')

# --- SECTION 1: THE PROBLEM (LEFT) ---
# Legacy AI
draw_box(ax, 1, 6.5, 3.5, 2, c_legacy, 'Deployed Legacy AI', 
         ['Trained on Old Dataset', '✓ Correctly detects Disease X'], '#1565C0')
# Insert your local image
ax.imshow(eye_img, extent=[1.5, 4.5, 4.5, 6.3], zorder=3, aspect='auto')

# Updated AI
draw_box(ax, 1, 2, 3.5, 2, c_update, 'New Updated AI', 
         ['Trained on New Dataset', '✗ Misses Disease X (Regression)'], '#C62828')
# Insert your local image
ax.imshow(eye_img, extent=[1.5, 4.5, 0.2, 1.8], zorder=3, aspect='auto')
ax.text(2.75, 1.0, '✗ Missing finding', color='#C62828', fontsize=14, fontweight='bold', ha='center', zorder=4,
        path_effects=[withStroke(linewidth=3, foreground='white')])

draw_arrow(ax, 3.5, 6.4, 3.5, 4.2) # Arrow from Legacy to Updated

# --- SECTION 2: CURE METHODOLOGY (CENTER) ---
# Main CURE Box
cure_rect = patches.FancyBboxPatch((5.5, 1.5), 6.5, 7.5, boxstyle="round,pad=0.3", 
                                   fc='#FAFAFA', ec='#2E7D32', lw=3, linestyle='--', zorder=2)
ax.add_patch(cure_rect)
ax.text(8.75, 8.6, 'CURE (Section 3 Methodology)', ha='center', va='center', 
        fontsize=16, fontweight='bold', color='#2E7D32', zorder=3)

# Phase 1
draw_box(ax, 6, 6.8, 5.5, 1.3, c_cure, 'Phase 1: Congruence Constraint', 
         ['Anchor thresholds to keep all findings', 'the legacy AI correctly detected.'], '#2E7D32')

# Phase 2
draw_box(ax, 6, 4.8, 5.5, 1.3, c_cure, 'Phase 2: Legacy Anchoring', 
         ['Estimate threshold shape using the', 'larger, labeled legacy validation data.'], '#2E7D32')

# Phase 3
draw_box(ax, 6, 2.8, 5.5, 1.3, c_cure, 'Phase 3: Certified Recalibration', 
         ['Apply one global shift. Verify on new data', 'that risk of missing findings is below limit.'], '#2E7D32')

# Arrows between phases
draw_arrow(ax, 8.75, 6.7, 8.75, 6.2, color='#2E7D32')
draw_arrow(ax, 8.75, 4.7, 8.75, 4.2, color='#2E7D32')

# Arrow from Legacy AI to CURE Phase 1
draw_arrow(ax, 4.6, 7.5, 5.9, 7.5, color='#1565C0')
ax.text(5.25, 7.7, "Extract\nfindings", ha='center', va='bottom', fontsize=10, color='#1565C0')

# Arrow from Updated AI to CURE Phase 3
draw_arrow(ax, 4.6, 3.0, 5.9, 3.4, color='#C62828')

# --- SECTION 3: THE SOLUTION (RIGHT) ---
draw_box(ax, 13, 4.5, 4, 2, c_output, 'Final Certified AI', 
         ['✓ Retains old findings (Guaranteed)', '✓ Accepts new dataset knowledge', '✓ No retraining required'], '#6A1B9A')
# Insert your local image
ax.imshow(eye_img, extent=[13.5, 16.5, 2.5, 4.3], zorder=3, aspect='auto')
ax.text(15, 3.5, '✓ Finding saved', color='#2E7D32', fontsize=14, fontweight='bold', ha='center', zorder=4,
        path_effects=[withStroke(linewidth=3, foreground='white')])

# Arrow from CURE to Final AI
draw_arrow(ax, 12.1, 5.5, 12.9, 5.5, color='#2E7D32')

# Add a legend for the eyes
ax.text(8.75, 0.8, "Visual Grounding: Retinal Fundus Images (Eyes) are processed to detect multiple diseases.", 
        ha='center', va='center', fontsize=11, style='italic', color='#555555')

plt.tight_layout()
plt.savefig('CURE_Methodology_Pipeline.png', dpi=300, bbox_inches='tight')
plt.show()