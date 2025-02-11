from fpdf import FPDF

# Add a Unicode font (DejaVuSans) to parse utf-8 correctly
# pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
# pdf.set_font('DejaVu', '', 12)
# pdf.cell(200, 10, txt="Pay-as-you-go (more cost control)", ln=True)

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        self.set_font("Arial", style="B", size=16)
        self.cell(200, 10, "AWS vs. Vercel - Next.js Hosting Comparison", ln=True, align="C")
        self.ln(10)

    def multi_cell_table(self, data, col_widths):
        """
        Create a table with dynamic row heights.
        
        :param data: List of lists containing table data.
        :param col_widths: List of column widths.
        """
        self.set_font("Arial", size=12)
        num_cols = len(col_widths)

        # Determine row heights dynamically
        row_heights = []
        for row in data:
            line_heights = [self.get_string_width(str(cell)) // col_widths[idx] + 1 for idx, cell in enumerate(row)]
            row_heights.append(max(line_heights) * 8)  # Multiply by line height (8) for spacing

        # Print the table
        for i, row in enumerate(data):
            max_height = row_heights[i]
            for j, cell in enumerate(row):
                self.multi_cell(col_widths[j], 8, str(cell), border=1, align="C")
                x = self.get_x()
                y = self.get_y()
                self.set_xy(x + col_widths[j], y - max_height)
            self.ln(max_height)

    def table(self, title, headers, data):
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

    def center_title(self, title):
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, title, ln=True, align="C")
        pdf.ln(10)

    def sub_heading(self, sub_title, size=14, space=5):
        pdf.set_font("Arial", style="B", size=size)
        pdf.cell(200, 10, sub_title, ln=True)
        pdf.ln(space)

    def sub_heading_p(self, sub_title, paragraph):
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

    def title_bullet_points(self, arr, space=10):
        for title, points in arr:
            pdf.set_font("Arial", size=12)
            pdf.cell(space/2)  # Indentation
            pdf.cell(0, 8, title, ln=True)
            for point in points:
                pdf.cell(20)  # More indentation for sub-bullets
                pdf.cell(0, 8, f"  * {point}", ln=True)
                pdf.ln(space/4)
        pdf.ln(space)

    def bullet_points(self, arr, space=10):
        for point in arr:
            pdf.cell(20)  # More indentation for sub-bullets
            pdf.cell(0, 8, f"  * {point}", ln=True)
            pdf.ln(space/4)
        pdf.ln(space)

# Recreate with proper encoding for bullet points
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=10)
pdf.add_page()



# Save PDF
pdf_filename = "./AI_Book_Swapping_Platform.pdf"
pdf.output(pdf_filename)

pdf_filename
