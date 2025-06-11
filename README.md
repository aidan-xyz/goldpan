# Membership Segmentation and Order Value Processing Tool

## Overview

This software is a Flask-based web application that processes membership and order data files (Excel format) to deduplicate records, enrich customer data with order value information, categorize membership statuses, and segment customers into actionable categories for CRM systems like HubSpot.

## What the Software Does

* **Deduplicates Membership Records:**

  * Identifies and removes redundant memberships based on customer name and date logic (e.g., memberships that renew within 7 days of expiration are consolidated).

* **Adds Order Value Tiers:**

  * Merges customer membership data with order histories using customer IDs.
  * Calculates total order values per customer.
  * Estimates customer savings based on a percentage of their total order value.

* **Categorizes Memberships:**

  * Segments memberships by expiration status: Expired, Expiring Soon, or Active.

* **Segments Customers:**

  * Uses logic based on enrollment status, recent renewals, expiration proximity, and order value to assign customers into meaningful segments.

* **Exports Clean Data:**

  * Supports exporting files in standard Excel format or in HubSpot-ready CSV format.

* **HubSpot-Friendly Boolean Columns:**

  * Creates a separate column for each segment as a boolean (True/False), enabling seamless import and filtering within HubSpot.

## How It Works

1. **File Upload:**

   * Users upload two Excel files: one with membership data, and one with order histories.

2. **Deduplication:**

   * The system compares created and expiration dates to remove redundant memberships.

3. **Order Processing:**

   * Orders are summed per customer ID to calculate total purchase values.

4. **Categorization:**

   * Memberships are categorized based on days until expiration.

5. **Segmentation:**

   * Customers are segmented into multiple categories, each reflected as a boolean column.

6. **Export:**

   * Users can download the processed file in their chosen format, ready for CRM or further analysis.

## Why This is Useful

* **Streamlines CRM Segmentation:**

  * Automatically prepares data for import into HubSpot, saving manual work.

* **Improves Customer Targeting:**

  * Provides clear segments for tailored marketing and retention campaigns.

* **Eliminates Duplicate Records:**

  * Ensures clean, accurate customer lists.

* **Automates Order Value Enrichment:**

  * Adds valuable customer spending insights for prioritization.

* **Supports Smart Workflows:**

  * Boolean columns integrate perfectly with HubSpot list-building and workflow automation tools.

* **Highly Customizable:**

  * Can easily adapt to future segmentation criteria or CRM export formats.

---

This tool is particularly well-suited for sales and marketing teams aiming to:

* Launch loyalty programs
* Automate renewal workflows
* Analyze customer value tiers
* Improve retention strategies

The boolean segmentation workaround provides a robust, scalable solution that integrates cleanly with CRM systems like HubSpot without requiring additional parsing or data manipulation post-import.
