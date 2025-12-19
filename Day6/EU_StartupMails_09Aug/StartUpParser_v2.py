import json
import re
import psycopg2
from datetime import datetime
from dateutil import parser as date_parser

class StartupFundingParser:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        
    def connect_to_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            print("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("🔌 Database connection closed")
    
    def parse_funding_rounds(self, email_body, email_date):
        """Extract funding information from email body text"""
        funding_rounds = []
        
        # Debug: Print a portion of email body to see the format
        print(f"   📝 Email body sample: {email_body[:200]}...")
        
        # First, clean up markdown links [CompanyName](URL) -> CompanyName
        cleaned_body = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', email_body)
        
        # Pattern to match funding rounds like:
        # "Berlin-based Talon.One just raised €114 million to power..."
        # "Dutch startup Dexter Energy raised €23 million to scale..."
        
        patterns = [
            # Pattern 1: Location-based CompanyName raised €Amount
            r'([A-Za-z\-]+(?:\s+[A-Za-z\-]+)*)-based\s+([A-Za-z0-9\.\s]+?)\s+(?:just\s+)?(?:raised|secured|landed)\s+€([\d,\.]+)\s+(million|thousand|billion|K)(?:\s+to\s+(.+?)(?:\.|$))?',
            
            # Pattern 2: Location startup CompanyName raised €Amount
            r'([A-Za-z\-]+(?:\s+[A-Za-z\-]+)*)\s+startup\s+([A-Za-z0-9\.\s]+?)\s+(?:raised|secured|landed)\s+€([\d,\.]+)\s+(million|thousand|billion|K)(?:\s+to\s+(.+?)(?:\.|$))?',
            
            # Pattern 3: CompanyName (Location) raised €Amount  
            r'([A-Za-z0-9\.\s]+?)\s+\([A-Za-z\-\s,]+\)\s+(?:raised|secured|landed)\s+€([\d,\.]+)\s+(million|thousand|billion|K)(?:\s+to\s+(.+?)(?:\.|$))?'
        ]
        
        for i, pattern in enumerate(patterns, 1):
            print(f"   🔍 Testing pattern {i}...")
            matches = re.finditer(pattern, cleaned_body, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                try:
                    groups = match.groups()
                    
                    # Parse based on pattern structure
                    if i == 1:  # Location-based Company pattern
                        location = groups[0].strip()
                        company_name = groups[1].strip()
                        amount_str = groups[2].replace(',', '.')
                        scale = groups[3].lower()
                        description = groups[4] if len(groups) > 4 and groups[4] else ""
                    elif i == 2:  # Location startup Company pattern
                        location = groups[0].strip()
                        company_name = groups[1].strip()  
                        amount_str = groups[2].replace(',', '.')
                        scale = groups[3].lower()
                        description = groups[4] if len(groups) > 4 and groups[4] else ""
                    elif i == 3:  # Company (Location) pattern
                        company_name = groups[0].strip()
                        amount_str = groups[1].replace(',', '.')
                        scale = groups[2].lower()
                        description = groups[3] if len(groups) > 3 and groups[3] else ""
                        location = "Unknown"
                    
                    # Convert amount to EUR
                    amount = float(amount_str)
                    if scale.lower() == 'million':
                        amount_eur = amount * 1000000
                    elif scale.lower() == 'billion':
                        amount_eur = amount * 1000000000
                    elif scale.lower() == 'thousand':
                        amount_eur = amount * 1000
                    elif scale.lower() == 'k':
                        amount_eur = amount * 1000
                    else:
                        amount_eur = amount
                    
                    # Clean company name (remove brackets if present)
                    company_name = re.sub(r'[\[\]]', '', company_name).strip()
                    
                    # Try to extract country from location
                    country = self.extract_country(location)
                    
                    funding_round = {
                        'company_name': company_name,
                        'location': location,
                        'country': country,
                        'amount_eur': amount_eur,
                        'amount_original': amount,
                        'original_currency': 'EUR',
                        'description': description.strip() if description else "",
                        'newsletter_date': email_date,
                        'round_type': self.guess_round_type(amount_eur, description)
                    }
                    
                    funding_rounds.append(funding_round)
                        
                except (ValueError, IndexError) as e:
                    print(f"⚠️  Error parsing match: {match.group(0)[:50]}... - {e}")
                    continue
        
        return funding_rounds
    
    def extract_country(self, location):
        """Extract country from location string"""
        location_lower = location.lower()
        
        # Common European cities/countries mapping
        country_mapping = {
            'berlin': 'Germany', 'munich': 'Germany', 'hamburg': 'Germany',
            'london': 'United Kingdom', 'manchester': 'United Kingdom',
            'paris': 'France', 'lyon': 'France', 'marseille': 'France',
            'amsterdam': 'Netherlands', 'rotterdam': 'Netherlands', 'dutch': 'Netherlands',
            'stockholm': 'Sweden', 'gothenburg': 'Sweden', 'swedish': 'Sweden',
            'copenhagen': 'Denmark', 'danish': 'Denmark',
            'oslo': 'Norway', 'norwegian': 'Norway',
            'helsinki': 'Finland', 'finnish': 'Finland',
            'madrid': 'Spain', 'barcelona': 'Spain', 'spanish': 'Spain',
            'rome': 'Italy', 'milan': 'Italy', 'italian': 'Italy',
            'prague': 'Czech Republic', 'czech': 'Czech Republic',
            'vienna': 'Austria', 'austrian': 'Austria',
            'zurich': 'Switzerland', 'geneva': 'Switzerland', 'swiss': 'Switzerland',
            'brussels': 'Belgium', 'belgian': 'Belgium',
            'lisbon': 'Portugal', 'porto': 'Portugal', 'portuguese': 'Portugal',
            'athens': 'Greece', 'greek': 'Greece',
            'warsaw': 'Poland', 'krakow': 'Poland', 'polish': 'Poland',
            'katowice': 'Poland'
        }
        
        for key, country in country_mapping.items():
            if key in location_lower:
                return country
        
        return location  # Return original if no mapping found
    
    def guess_round_type(self, amount_eur, description):
        """Guess the round type based on amount and description"""
        description_lower = description.lower()
        
        # Check for explicit round mentions
        if 'seed' in description_lower:
            return 'Seed'
        elif 'series a' in description_lower:
            return 'Series A'
        elif 'series b' in description_lower:
            return 'Series B'
        elif 'series c' in description_lower:
            return 'Series C'
        
        # Guess based on amount (rough European startup standards)
        if amount_eur < 1000000:  # < €1M
            return 'Pre-Seed'
        elif amount_eur < 5000000:  # €1M - €5M
            return 'Seed'
        elif amount_eur < 20000000:  # €5M - €20M
            return 'Series A'
        elif amount_eur < 50000000:  # €20M - €50M
            return 'Series B'
        else:  # €50M+
            return 'Series C+'
    
    def insert_company(self, company_data):
        """Insert company into database, return company_id"""
        cursor = self.conn.cursor()
        
        # Check if company already exists
        cursor.execute(
            "SELECT id FROM companies WHERE LOWER(name) = LOWER(%s)",
            (company_data['company_name'],)
        )
        
        existing = cursor.fetchone()
        if existing:
            cursor.close()
            return existing[0]
        
        # Insert new company
        insert_query = """
        INSERT INTO companies (name, country, city, description, created_at)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        
        cursor.execute(insert_query, (
            company_data['company_name'],
            company_data['country'],
            company_data['location'],
            company_data['description'][:500],  # Limit description length
            datetime.now()
        ))
        
        company_id = cursor.fetchone()[0]
        cursor.close()
        return company_id
    
    def insert_funding_round(self, company_id, funding_data):
        """Insert funding round into database"""
        cursor = self.conn.cursor()
        
        insert_query = """
        INSERT INTO funding_rounds (
            company_id, round_type, amount_eur, amount_original, 
            original_currency, announced_date, newsletter_date, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        cursor.execute(insert_query, (
            company_id,
            funding_data['round_type'],
            funding_data['amount_eur'],
            funding_data['amount_original'],
            funding_data['original_currency'],
            funding_data['newsletter_date'],
            funding_data['newsletter_date'],
            datetime.now()
        ))
        
        funding_id = cursor.fetchone()[0]
        cursor.close()
        return funding_id
    
    def insert_investor(self, name):
        """Insert or get existing investor"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM investors WHERE LOWER(name) = LOWER(%s)",
            (name,)
        )
        existing = cursor.fetchone()
        if existing:
            cursor.close()
            return existing[0]
        cursor.execute("""
            INSERT INTO investors (name, created_at) VALUES (%s, %s)
            RETURNING id
        """, (name[:255], datetime.now()))
        investor_id = cursor.fetchone()[0]
        cursor.close()
        return investor_id
    
    def insert_newsletter(self, subject, date_received, message_id=None, processed=False):
        """Insert newsletter entry and return its ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO newsletters (subject, date_received, date_processed, message_id, processed)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (
            subject[:500],             # subject column limit from schema (500)
            date_received,              # date_received column
            datetime.now(),              # date_processed set to current time
            message_id[:255] if message_id else None,  # optional message_id
            processed                   # defaults to False
        ))
        newsletter_id = cursor.fetchone()[0]
        cursor.close()
        return newsletter_id

    
    def extract_investors(self, text):
        # Basic pattern to extract investor names from phrases like "backed by Tiger Global and Accel"
        patterns = [
            r'backed by\s+([A-Z][A-Za-z0-9\s,&\-]+)',
            r'investment from\s+([A-Z][A-Za-z0-9\s,&\-]+)',
            r'funding from\s+([A-Z][A-Za-z0-9\s,&\-]+)',
            r'led by\s+([A-Z][A-Za-z0-9\s,&\-]+)'
        ]
        investors = []
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                for match in matches:
                    # Split multiple investors separated by "and" or commas
                    parts = re.split(r'\sand\s|,', match)
                    for part in parts:
                        inv = part.strip()
                        if inv and inv not in investors:
                            investors.append(inv)
        return investors
    
    def link_funding_investor(self, funding_id, investor_id):
        """Link investor to funding round"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO funding_investors (funding_round_id, investor_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
        """, (funding_id, investor_id))
        cursor.close()

    def extract_investors(self, text):
        # Basic pattern to extract investor names from phrases like "backed by Tiger Global and Accel"
        patterns = [
            r'backed by\s+([A-Z][A-Za-z0-9\s,&\-]+)',
            r'investment from\s+([A-Z][A-Za-z0-9\s,&\-]+)',
            r'funding from\s+([A-Z][A-Za-z0-9\s,&\-]+)',
            r'led by\s+([A-Z][A-Za-z0-9\s,&\-]+)'
        ]
        investors = []
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                for match in matches:
                    # Split multiple investors separated by "and" or commas
                    parts = re.split(r'\sand\s|,', match)
                    for part in parts:
                        inv = part.strip()
                        if inv and inv not in investors:
                            investors.append(inv)
        return investors
    
    def process_emails_json(self, json_file_path):
        """Process emails and populate companies, funding, newsletters, investors, and links"""
        print(f"📧 Loading emails from {json_file_path}")
    
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                emails = json.load(f)
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return
    
        # Auto-detect VARCHAR limits from DB schema
        def get_column_limits(table_name):
            query = """
            SELECT column_name, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = %s
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (table_name,))
            limits = {}
            for col, max_len in cursor.fetchall():
                limits[col] = max_len
            cursor.close()
            return limits
    
        companies_limits = get_column_limits("companies")
        funding_limits = get_column_limits("funding_rounds")
        newsletters_limits = get_column_limits("newsletters")
        investors_limits = get_column_limits("investors")
    
        print(f"🔍 Column length limits - companies: {companies_limits}")
        print(f"🔍 Column length limits - funding_rounds: {funding_limits}")
        print(f"🔍 Column length limits - newsletters: {newsletters_limits}")
        print(f"🔍 Column length limits - investors: {investors_limits}")
    
        print(f"📊 Found {len(emails)} emails to process")
        total_funding_rounds = 0
    
        for i, email in enumerate(emails, 1):
            print(f"\n📧 Processing email {i}/{len(emails)}")
            print(f"   Subject: {email['subject'][:50]}...")
    
            try:
                email_date = date_parser.parse(email['date']).date()
            except:
                email_date = datetime.now().date()
    
            # Insert newsletter record
            try:
                # Truncate newsletter fields respecting DB limits
                subject_trunc = email['subject'][:newsletters_limits.get('subject', 255)] if newsletters_limits.get('subject') else email['subject']
                newsletter_date = email_date
    
                newsletter_id = self.insert_newsletter(subject_trunc,email_date,message_id=email.get('message_id'),processed=False)
                
            except Exception as e:
                print(f"❌ Failed to insert newsletter for email {i}: {e}")
                continue
    
            funding_rounds = self.parse_funding_rounds(email['body'], email_date)
            print(f"   Found {len(funding_rounds)} funding rounds")
    
            # Extract investors once per email body (you might also link specific investors to specific funding if you have those details)
            investors_names = self.extract_investors(email['body'])
            investor_ids = []
            for inv_name in investors_names:
                try:
                    inv_trunc = inv_name[:investors_limits.get('name', 255)] if investors_limits.get('name') else inv_name
                    inv_id = self.insert_investor(inv_trunc)
                    investor_ids.append(inv_id)
                except Exception as e:
                    print(f"   ❌ Failed to insert investor '{inv_name}': {e}")
                    continue
    
            for funding in funding_rounds:
                try:
                    # Attach newsletter ID to funding
                    funding['newsletter_id'] = newsletter_id
    
                    # Truncate company/funding fields based on detected limits
                    if companies_limits.get("name"):
                        funding['company_name'] = funding['company_name'][:companies_limits["name"]]
                    if companies_limits.get("country"):
                        funding['country'] = funding['country'][:companies_limits["country"]]
                    if companies_limits.get("city"):
                        funding['location'] = funding['location'][:companies_limits["city"]]
                    if companies_limits.get("description"):
                        funding['description'] = funding['description'][:companies_limits["description"]]
    
                    if funding_limits.get("round_type"):
                        funding['round_type'] = funding['round_type'][:funding_limits["round_type"]]
    
                    # Insert company and funding round
                    company_id = self.insert_company(funding)
                    funding_id = self.insert_funding_round(company_id, funding)
    
                    # Link all investors found in this email to this funding round
                    for inv_id in investor_ids:
                        self.link_funding_investor(funding_id, inv_id)
    
                    self.conn.commit()
                    print(f"   ✅ {funding['company_name']} - €{funding['amount_eur']:,.0f}")
                    total_funding_rounds += 1
    
                except Exception as e:
                    self.conn.rollback()
                    print(f"   ❌ Error inserting funding '{funding.get('company_name', 'Unknown')}': {e}")
                    continue
    
        print(f"\n🎉 Import complete! Added {total_funding_rounds} funding rounds linked to newsletters and investors")

def main():
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'EU_Startups',
        'user': 'postgres',
        'password': 'Shubham2100'  # Replace with your actual password
    }
    
    # Create parser instance
    parser = StartupFundingParser(db_config)
    
    # Connect to database
    if not parser.connect_to_db():
        return
    
    try:
        # Process the JSON file
        parser.process_emails_json('emails.json')  # Update path if needed
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    
    finally:
        parser.close_connection()

if __name__ == "__main__":
    main()