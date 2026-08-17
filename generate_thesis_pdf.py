from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
import os

def create_pdf(output_filename):
    doc = SimpleDocTemplate(output_filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='CenterTitle', alignment=TA_CENTER, fontSize=16, spaceAfter=20, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER, fontSize=12, spaceAfter=10))
    styles.add(ParagraphStyle(name='CustomHeading', fontSize=14, spaceAfter=10, spaceBefore=15, fontName="Helvetica-Bold"))
    
    Story = []
    
    # Title
    Story.append(Paragraph("Treadmill Offset Calibration and Modeling Report", styles['CenterTitle']))
    Story.append(Paragraph("A Methodological Overview for Thesis Documentation", styles['Center']))
    Story.append(Spacer(1, 30))
    
    # Section 1
    Story.append(Paragraph("1. Objective and Problem Statement", styles['CustomHeading']))
    text1 = ("Ground reaction forces (GRFs) collected from an instrumented treadmill often exhibit baseline "
             "offsets even when the treadmill is empty. These offsets are not static; they fluctuate depending "
             "on the mechanical state of the treadmill, specifically the belt speed and the incline (slope) of the machine. "
             "Failing to correct for these dynamic offsets propagates errors into the Inverse Dynamics (ID) pipeline, "
             "resulting in inaccurate joint moment and joint power calculations. The objective of this methodology was to "
             "quantify these empty-treadmill offsets across a wide range of operational conditions and develop a robust, "
             "dynamic correction model to tare the force plates accurately during experimental trials.")
    Story.append(Paragraph(text1, styles['Justify']))
    
    # Section 2
    Story.append(Paragraph("2. Data Collection and Pre-processing", styles['CustomHeading']))
    text2 = ("Calibration data was collected by running the empty treadmill across a dense grid of conditions: "
             "15 different speeds (ranging from 0.2 mph to 1.5 mph) at 6 different slopes (0%, 2.1%, 2.5%, 3.1%, 3.5%, 4.0%, and 4.5%). "
             "For each condition, a .mot file was recorded containing the forces for Plate 4 (Left) and Plate 5 (Right). "
             "A data parsing script was developed to iterate through all 101 recorded .mot files. For each file, the median "
             "and mean values were computed over time for the six relevant force channels: Fx (medial-lateral shear), "
             "Fy (vertical), and Fz (anterior-posterior shear). The median was selected as the primary metric to "
             "ensure robustness against transient mechanical vibrations or noise spikes.")
    Story.append(Paragraph(text2, styles['Justify']))
    
    # Section 3
    Story.append(Paragraph("3. Linear Modeling and Analysis", styles['CustomHeading']))
    text3 = ("An initial hypothesis posited that the relationship between the machine's offset, speed, and slope could be "
             "described using a Multiple Linear Regression model (Offset = B0 + B1*Speed + B2*Slope). We fit independent OLS "
             "regression models for each force axis. The analysis yielded the following key insights concerning the coordinate system:")
    Story.append(Paragraph(text3, styles['Justify']))
    Story.append(Spacer(1, 10))
    
    # Bullet points
    bullets = [
        "X-Axis (Medial-Lateral Shear): Displayed a very strong linear relationship with speed and slope, with R-squared values reaching approximately 0.83.",
        "Y-Axis (Vertical Force): Exhibited a moderate linear relationship, with R-squared values ranging between 0.30 and 0.54.",
        "Z-Axis (Anterior-Posterior Shear): Showed an extremely poor linear fit, with R-squared values falling below 0.02. A simple linear plane could not explain the variance in these specific offsets."
    ]
    for bullet in bullets:
        Story.append(Paragraph(f"• {bullet}", styles['Justify']))
        Story.append(Spacer(1, 5))
        
    text4 = ("Because the linear models failed to reliably predict the vertical and anterior-posterior shear forces, "
             "a unified linear approach was rejected in favor of a non-linear, multidimensional model.")
    Story.append(Paragraph(text4, styles['Justify']))
    
    # Section 4
    Story.append(Paragraph("4. Final Model Selection: 2D Grid Interpolation", styles['CustomHeading']))
    text5 = ("Given the density of the calibration grid, the optimal solution implemented was a 2D Grid Interpolation "
             "(Bivariate Spline / Lookup Table) approach. Six completely independent interpolator models were constructed "
             "(one for each force axis per plate) using Scipy's LinearNDInterpolator. "
             "This approach avoids forcing a specific mathematical formula onto the complex mechanical behavior of the treadmill. "
             "Instead, it constructs a smooth, continuous 2D surface mathematically tying the empirical calibration points together. "
             "When querying the model for an experimental condition (e.g., Speed = 1.25 mph, Slope = 3.3%), the interpolator "
             "procedurally estimates a highly accurate offset by weighting the nearest known calibration medians.")
    Story.append(Paragraph(text5, styles['Justify']))
    
    # Section 5
    Story.append(Paragraph("5. Implementation and Pipeline Integration", styles['CustomHeading']))
    text6 = ("A reusable Python class, TreadmillOffsetCorrector, was developed to encapsulate this logic. "
             "Integrated directly into the ID computing pipeline, this module dynamically tares the force plate data array "
             "by subtracting the interpolated baseline offsets before the data is handed to the OpenSim Inverse Dynamics solver. "
             "This procedural correction ensures that the resultant joint power calculations are untainted by machine-specific baseline shifts.")
    Story.append(Paragraph(text6, styles['Justify']))
    
    # Image
    img_path = r"d:\AFO_Codes\TreadmillOffset\plot_3d_ground_force4_vx_median.png"
    if os.path.exists(img_path):
        Story.append(Spacer(1, 20))
        Story.append(Paragraph("Figure 1: 3D Visualization of Speed, Slope, and Force Offset with Regression Plane", styles['Center']))
        Story.append(Spacer(1, 10))
        img = Image(img_path, width=400, height=320)
        Story.append(img)
    
    doc.build(Story)
    print(f"PDF generated successfully at {output_filename}")

if __name__ == "__main__":
    output_path = r"d:\AFO_Codes\Treadmill_Offset_Methodology_Report.pdf"
    create_pdf(output_path)
