#!/usr/bin/env python3
"""
Generate a benchmark CSV file with 50 rows for testing classification models.
Categories: Technical Support, Billing, Product Feedback, Account Management, General Inquiry
"""

import csv

data = [
    # Technical Support (10 rows)
    ["1", "My application crashes every time I try to upload a file larger than 10MB", "Technical Support"],
    ["2", "I'm getting a 500 error when accessing the API endpoint", "Technical Support"],
    ["3", "The software won't install on Windows 11, shows compatibility error", "Technical Support"],
    ["4", "Login page is not loading, stuck on white screen", "Technical Support"],
    ["5", "Database connection keeps timing out after 30 seconds", "Technical Support"],
    ["6", "Mobile app freezes when I try to export data", "Technical Support"],
    ["7", "Unable to connect to VPN, authentication fails", "Technical Support"],
    ["8", "Getting SSL certificate error on the dashboard", "Technical Support"],
    ["9", "Reports are generating blank PDFs", "Technical Support"],
    ["10", "Two-factor authentication codes not arriving", "Technical Support"],
    
    # Billing (10 rows)
    ["11", "I was charged twice for my monthly subscription", "Billing"],
    ["12", "Need to update my credit card information", "Billing"],
    ["13", "Can I get a refund for the annual plan?", "Billing"],
    ["14", "Invoice #12345 shows incorrect amount", "Billing"],
    ["15", "My payment was declined but I have sufficient funds", "Billing"],
    ["16", "How do I upgrade to the enterprise plan?", "Billing"],
    ["17", "Cancel my subscription effective immediately", "Billing"],
    ["18", "Need a receipt for last month's payment", "Billing"],
    ["19", "What payment methods do you accept?", "Billing"],
    ["20", "Proration charges don't match what was quoted", "Billing"],
    
    # Product Feedback (10 rows)
    ["21", "Love the new dark mode feature! Very easy on the eyes", "Product Feedback"],
    ["22", "The search function could be faster", "Product Feedback"],
    ["23", "Please add support for CSV exports", "Product Feedback"],
    ["24", "The UI is confusing, needs better navigation", "Product Feedback"],
    ["25", "Excellent customer service experience today", "Product Feedback"],
    ["26", "Would be great to have keyboard shortcuts", "Product Feedback"],
    ["27", "The mobile app needs a tablet optimized version", "Product Feedback"],
    ["28", "Integration with Slack would be amazing", "Product Feedback"],
    ["29", "Dashboard loads slowly with large datasets", "Product Feedback"],
    ["30", "Really appreciate the new bulk editing feature", "Product Feedback"],
    
    # Account Management (10 rows)
    ["31", "How do I reset my password?", "Account Management"],
    ["32", "Need to add three new users to my team account", "Account Management"],
    ["33", "Can't remember my username", "Account Management"],
    ["34", "Want to change my email address on file", "Account Management"],
    ["35", "How do I delete my account permanently?", "Account Management"],
    ["36", "Need to transfer ownership to another admin", "Account Management"],
    ["37", "How do I set up single sign-on for my organization?", "Account Management"],
    ["38", "Remove user access for former employee", "Account Management"],
    ["39", "Update company name and address on account", "Account Management"],
    ["40", "Enable two-factor authentication for all users", "Account Management"],
    
    # General Inquiry (10 rows)
    ["41", "What are your business hours?", "General Inquiry"],
    ["42", "Do you have an office in Europe?", "General Inquiry"],
    ["43", "Is there a student discount available?", "General Inquiry"],
    ["44", "What's the difference between Pro and Enterprise plans?", "General Inquiry"],
    ["45", "Do you offer training webinars?", "General Inquiry"],
    ["46", "Are you hiring software engineers?", "General Inquiry"],
    ["47", "What's your data retention policy?", "General Inquiry"],
    ["48", "Is the service GDPR compliant?", "General Inquiry"],
    ["49", "Where can I find the API documentation?", "General Inquiry"],
    ["50", "Do you have a referral program?", "General Inquiry"],
]

# Write to CSV
with open('benchmark_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'category'])
    writer.writerows(data)

print("✓ Created benchmark_data.csv with 50 rows")
print("✓ Categories: Technical Support, Billing, Product Feedback, Account Management, General Inquiry")