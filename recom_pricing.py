from fpdf import FPDF

#
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=10)
pdf.add_page()

from pdf import table, center_title, sub_heading, sub_heading_p, bullet_points, title_bullet_points, table_multi

# Service Costs: Estimated for Project Duration (3 to 5 Months)
center_title(pdf, "Service Costs: Estimated for Project Duration (3 to 5 Months)")

# backend recommendation
sub_heading(pdf, "1. Node.js vs. Django: Which Backend is Better Recommended", 10, 5)
table_multi(pdf, "Backend Recommendation", ["Feature", "Node.js (Express.js)", "Django (Python)"], [
    ["Speed", "Faster (non-blocking, event-driven)", "Slower (synchronous but optimized for security)"],
    ["Performance", "High concurrency, handles real-time operations well", "Great for CPU-heavy tasks, but slower for async operations"],
    ["Ease of Use", "Best for JavaScript devs (same language as frontend)", "Best for Python devs, includes built-in admin panel"],
    ["Database Compatibility", "MongoDB, PostgreSQL, MySQL", "PostgreSQL, MySQL, SQLite"],
    ["Microservices-Friendly?", "Yes (easily integrates with multiple services)", "Not the best, more monolithic"],
    ["Security", "Secure with middleware but needs extra setup", "Highly secure (built-in protections)"],
    ["AI Integration", "Needs Python-based microservices for AI", "Better for AI (since AI models are in Python)"],
    ["When to Use", "Real-time, async, scalable apps", "Secure, structured apps"]
])

# Backend recommendation
sub_heading(pdf, "Recommendation: Use Node.js", 10, 5)
bullet_points(pdf, [
    "Node.js is recommended for this project due to its speed, performance, and microservices-friendly architecture.",
    "Node.js also integrates well with MongoDB, which is a popular choice for NoSQL databases.",
    "Works better with React (same JavaScript ecosystem).",
    "Handles real-time book swapping, chats, and notifications efficiently.",
    "AI models (Python) can be hosted as separate microservices and integrated via API calls.",
    "Scales better for high-concurrency requests (important for social features)."
], 5)

# database recommendation



# Save PDF
pdf_filename = "./Service_Recomendation_And_Pricing.pdf"
pdf.output(pdf_filename)

pdf_filename