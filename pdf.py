from fpdf import FPDF

# Recreate with proper encoding for bullet points
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=10)
pdf.add_page()

# Add a Unicode font (DejaVuSans) to parse utf-8 correctly
# pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
# pdf.set_font('DejaVu', '', 12)
# pdf.cell(200, 10, txt="Pay-as-you-go (more cost control)", ln=True)

def table(pdf, title, headers, data):
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

def table_multi(pdf, title, headers, data):
    # Add Section Heading
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, title, ln=True)
    pdf.ln(5)

    # Table Headers
    pdf.set_font("Arial", style="B", size=12)
    col_widths = [40, 75, 75]  # Define column widths
    row_height = 10  # Define row height
    current_height = 10

    # Draw headers
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], current_height, header, border=1, align="C")
    pdf.ln()

    # Table Data
    pdf.set_font("Arial", size=10)
    for row in data:
        # Track the maximum height of the row
        max_height = row_height

        # Calculate the height required for each cell
        for i, item in enumerate(row):
            # Get the number of lines needed for the text
            lines = pdf.multi_cell(col_widths[i], row_height, txt=item, border=0, align="C", split_only=True)
            cell_height = len(lines) * row_height
            if cell_height > max_height:
                max_height = cell_height

        # Draw the cells with the calculated height
        x = pdf.get_x()
        y = pdf.get_y()
        for i, item in enumerate(row):
            pdf.multi_cell(col_widths[i], row_height, txt=item, border=1, align="C")
            pdf.set_xy(x + col_widths[i], y)
            x += col_widths[i]
        current_height = max_height
        pdf.ln(max_height)

def center_title(pdf, title):
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, title, ln=True, align="C")
    pdf.ln(10)

def sub_heading(pdf, sub_title, size=14, space=5):
    pdf.set_font("Arial", style="B", size=size)
    pdf.cell(200, 10, sub_title, ln=True)
    pdf.ln(space)

def sub_heading_p(pdf, sub_title, paragraph):
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

def title_bullet_points(pdf, arr, space=10):
    for title, points in arr:
        pdf.set_font("Arial", size=12)
        pdf.cell(space/2)  # Indentation
        pdf.cell(0, 8, title, ln=True)
        for point in points:
            pdf.cell(20)  # More indentation for sub-bullets
            pdf.cell(0, 8, f"  * {point}", ln=True)
            pdf.ln(space/4)
    pdf.ln(space)

def bullet_points(pdf, arr, space=10):
    for point in arr:
        pdf.cell(20)  # More indentation for sub-bullets
        pdf.cell(0, 8, f"  * {point}", ln=True)
        pdf.ln(space/4)
    pdf.ln(space)



# Save PDF
pdf_filename = "./AI_Book_Swapping_Platform.pdf"
pdf.output(pdf_filename)

pdf_filename
