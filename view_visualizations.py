"""
Visualization Preview Generator

This script generates a simple HTML report with thumbnails of the transformer data visualizations
to make it easy to view all the visualizations in one place.
"""

import os
import base64
from PIL import Image
import io
import datetime

def generate_visualization_report(visualization_dirs=None, output_file=None):
    """
    Generate HTML report with thumbnails of visualizations
    
    Parameters:
    -----------
    visualization_dirs : list, optional
        List of directories containing visualizations. If None, use default directories.
    output_file : str, optional
        Path to save the HTML report. If None, a default path will be used.
    """
    if visualization_dirs is None:
        visualization_dirs = ['visualizations', 'visualizations/real_data']
    
    if output_file is None:
        output_file = f'transformer_visualizations_report_{datetime.datetime.now().strftime("%Y%m%d")}.html'
    
    # Generate date string for the report
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transformer Data Visualizations</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; margin-top: 30px; }}
        .visualization-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
        .visualization-item {{ max-width: 400px; }}
        .visualization-item img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        .visualization-caption {{ margin-top: 5px; font-size: 0.9em; color: #555; }}
    </style>
</head>
<body>
    <h1>Transformer Price Calculator Visualizations</h1>
    <p>Generated on: {date_str}</p>
"""
    
    # Process each visualization directory
    for viz_dir in visualization_dirs:
        if not os.path.exists(viz_dir):
            continue
            
        html_content += f'<h2>{os.path.basename(viz_dir) or "Main"} Visualizations</h2>\n'
        html_content += '<div class="visualization-container">\n'
        
        # Get all image files in the directory
        image_files = [f for f in os.listdir(viz_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        for img_file in image_files:
            img_path = os.path.join(viz_dir, img_file)
            
            try:
                # Create a thumbnail for the image
                with Image.open(img_path) as img:
                    # Resize the image while maintaining aspect ratio
                    img.thumbnail((400, 400))
                    
                    # Convert to base64 for embedding in HTML
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format=img.format)
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                    
                    # Add to HTML content
                    html_content += f"""<div class="visualization-item">
    <img src="data:image/{img.format.lower()};base64,{img_base64}" alt="{img_file}">
    <div class="visualization-caption">{img_file}</div>
</div>
"""
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        html_content += '</div>\n'
    
    # Close HTML content
    html_content += """</body>
</html>
"""
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated visualization report at {output_file}")
    return output_file

if __name__ == "__main__":
    report_file = generate_visualization_report()
    
    # Try to open the HTML file in the default browser
    try:
        import webbrowser
        webbrowser.open(report_file)
    except:
        pass 