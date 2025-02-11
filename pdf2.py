from fpdf import FPDF

# Create a PDF object
pdf = FPDF()

# Add the DejaVuSans font (regular and bold variants)
pdf.add_font('DejaVuSans', '', 'DejaVuSans.ttf', uni=True)  # Regular
# pdf.add_font('DejaVuSans', 'B', 'DejaVuSans-Bold.ttf', uni=True)  # Bold

# Set the default font
pdf.set_font('DejaVuSans', '', 12)

# Set auto page break
pdf.set_auto_page_break(auto=True, margin=10)
pdf.add_page()


def table(title, headers, data):
    # Add Section Heading
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, title, ln=True)
    pdf.ln(5)

    # Table Headers
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(40, 10, headers[0], border=1, align="C")
    for header in headers[1:]:
        pdf.cell(75, 10, header, border=1, align="C")
    pdf.ln()

    # Table Data
    pdf.set_font("Arial", size=10)
    for row in data:
        pdf.cell(40, 10, row[0], border=1, align="C")
        for item in row[1:]:
            pdf.cell(75, 10, txt=item, border=1, align="C")
        pdf.ln()

def center_title(title):
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, title, ln=True, align="C")
    pdf.ln(10)

def sub_heading(sub_title, size=14, space=5):
    pdf.set_font("Arial", style="B", size=size)
    pdf.cell(200, 10, sub_title, ln=True)
    pdf.ln(space)

def sub_heading_p(sub_title, paragraph):
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, sub_title, ln=True)
    pdf.ln(5)

    if isinstance(paragraph, list):
        for p in paragraph:
            text =''.join(p)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 8, p)
            pdf.ln(5)
    else:
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, paragraph)
        pdf.ln(10) 

def bullet_points(arr, space=10):
    for title, points in arr:
        pdf.set_font("Arial", size=12)
        pdf.cell(space/2)  # Indentation
        pdf.cell(0, 8, title, ln=True)
        for point in points:
            pdf.cell(20)  # More indentation for sub-bullets
            pdf.cell(0, 8, f"  * {point}", ln=True)
            pdf.ln(space/4)
    pdf.ln(space)



# Save PDF
pdf_filename = "./AI_Book_Swapping_Platform.pdf"
pdf.output(pdf_filename)

pdf_filename