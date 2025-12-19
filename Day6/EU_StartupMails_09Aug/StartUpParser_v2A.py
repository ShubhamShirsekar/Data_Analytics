import json
import re
import psycopg2
from datetime import datetime
from dateutil import parser as date_parser
from email.header import decode_header


class StartupFundingParser:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None

    def connect_to_db(self):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            print("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

    def close_connection(self):
        if self.conn:
            self.conn.close()
            print("🔌 Database connection closed")

    def clean_subject(self, raw_subject):
        """Decode and normalize subject line text."""
        if not raw_subject:
            return ""
        try:
            decoded_parts = decode_header(raw_subject)
            subject_parts = []
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    subject_parts.append(part.decode(encoding or "utf-8", errors="replace"))
                else:
                    subject_parts.append(part)
            clean_subj = " ".join(subject_parts).strip()
            return " ".join(clean_subj.split())
        except Exception:
            return raw_subject

    def _get_column_limits(self, table_name):
        query = """
        SELECT column_name, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = %s
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (table_name,))
        limits = {col: max_len for col, max_len in cursor.fetchall()}
        cursor.close()
        return limits

    def parse_funding_rounds(self, email_body, email_date):
        funding_rounds = []
        print(f"   📝 Email body sample: {email_body[:200]}...")
        cleaned_body = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', email_body)

        patterns = [
            r'([A-Za-z\-]+(?:\s+[A-Za-z\-]+)*)-based\s+([A-Za-z0-9\.\s]+?)\s+(?:just\s+)?(?:raised|secured|landed)\s+€([\d,\.]+)\s+(million|thousand|billion|K)(?:\s+to\s+(.+?)(?:\.|$))?',
            r'([A-Za-z\-]+(?:\s+[A-Za-z\-]+)*)\s+startup\s+([A-Za-z0-9\.\s]+?)\s+(?:raised|secured|landed)\s+€([\d,\.]+)\s+(million|thousand|billion|K)(?:\s+to\s+(.+?)(?:\.|$))?',
            r'([A-Za-z0-9\.\s]+?)\s+\([A-Za-z\-\s,]+\)\s+(?:raised|secured|landed)\s+€([\d,\.]+)\s+(million|thousand|billion|K)(?:\s+to\s+(.+?)(?:\.|$))?'
        ]

        for i, pattern in enumerate(patterns, 1):
            print(f"   🔍 Testing pattern {i}...")
            matches = re.finditer(pattern, cleaned_body, re.IGNORECASE | re.MULTILINE)

            for match in matches:
                try:
                    groups = match.groups()
                    if i == 1:
                        location, company_name, amount_str, scale, description = groups[0].strip(), groups[1].strip(), groups[2].replace(',', '.'), groups[3].lower(), groups[4] or ""
                    elif i == 2:
                        location, company_name, amount_str, scale, description = groups[0].strip(), groups[1].strip(), groups[2].replace(',', '.'), groups[3].lower(), groups[4] or ""
                    elif i == 3:
                        company_name, amount_str, scale, description = groups[0].strip(), groups[1].replace(',', '.'), groups[2].lower(), groups[3] or ""
                        location = "Unknown"

                    amount = float(amount_str)
                    if scale == 'million':
                        amount_eur = amount * 1_000_000
                    elif scale == 'billion':
                        amount_eur = amount * 1_000_000_000
                    elif scale in ['thousand', 'k']:
                        amount_eur = amount * 1_000
                    else:
                        amount_eur = amount

                    company_name = re.sub(r'[\[\]]', '', company_name).strip()
                    country = self.extract_country(location)

                    # find investors in description context
                    investors = self.extract_investors(description)

                    funding_rounds.append({
                        'company_name': company_name,
                        'location': location,
                        'country': country,
                        'amount_eur': amount_eur,
                        'amount_original': amount,
                        'original_currency': 'EUR',
                        'description': description.strip(),
                        'newsletter_date': email_date,
                        'round_type': self.guess_round_type(amount_eur, description),
                        'investors': investors
                    })

                except Exception as e:
                    print(f"⚠️  Error parsing match: {match.group(0)[:50]} - {e}")
                    continue

        return funding_rounds

    def extract_country(self, location):
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
            if key in location.lower():
                return country
        return location

    def guess_round_type(self, amount_eur, description):
        d = description.lower()
        if 'seed' in d: return 'Seed'
        if 'series a' in d: return 'Series A'
        if 'series b' in d: return 'Series B'
        if 'series c' in d: return 'Series C'
        if amount_eur < 1_000_000: return 'Pre-Seed'
        if amount_eur < 5_000_000: return 'Seed'
        if amount_eur < 20_000_000: return 'Series A'
        if amount_eur < 50_000_000: return 'Series B'
        return 'Series C+'

    def insert_company(self, data):
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM companies WHERE LOWER(name) = LOWER(%s)", (data['company_name'],))
        existing = cur.fetchone()
        if existing:
            cur.close()
            return existing[0]
        cur.execute("""
            INSERT INTO companies (name, country, city, description, created_at)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
        """, (data['company_name'], data['country'], data['location'], data['description'][:500], datetime.now()))
        cid = cur.fetchone()[0]
        cur.close()
        return cid

    def insert_funding_round(self, company_id, data):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO funding_rounds (
                company_id, round_type, amount_eur, amount_original,
                original_currency, announced_date, newsletter_date, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
        """, (company_id, data['round_type'], data['amount_eur'], data['amount_original'],
              data['original_currency'], data['newsletter_date'], data['newsletter_date'], datetime.now()))
        fid = cur.fetchone()[0]
        cur.close()
        return fid

    def insert_investor(self, name):
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM investors WHERE LOWER(name) = LOWER(%s)", (name,))
        existing = cur.fetchone()
        if existing:
            cur.close()
            return existing[0]
        cur.execute("INSERT INTO investors (name, created_at) VALUES (%s, %s) RETURNING id",
                    (name[:255], datetime.now()))
        iid = cur.fetchone()[0]
        cur.close()
        return iid

    def link_funding_investor(self, funding_id, investor_id):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO funding_investors (funding_round_id, investor_id)
            VALUES (%s, %s) ON CONFLICT DO NOTHING
        """, (funding_id, investor_id))
        cur.close()

    def insert_newsletter(self, subject, date_received, message_id=None, processed=False):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO newsletters (subject, date_received, date_processed, message_id, processed)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
        """, (subject[:500], date_received, datetime.now(),
              message_id[:255] if message_id else None, processed))
        nid = cur.fetchone()[0]
        cur.close()
        return nid

    def extract_investors(self, text):
        pattern = r'(?:backed by|led by|investment from|funding from|joined by|with participation from)\s+([A-Z][A-Za-z0-9\s\&\-\.,]+)'
        investors = []
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for match in matches:
            for part in re.split(r',| and ', match):
                inv = part.strip(" .")
                if len(inv) > 2 and not inv.lower().startswith("the ") and inv not in investors:
                    investors.append(inv)
        return investors

    def process_emails_json(self, json_file_path):
        print(f"📧 Loading emails from {json_file_path}")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                emails = json.load(f)
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return

        companies_limits = self._get_column_limits("companies")
        funding_limits = self._get_column_limits("funding_rounds")
        newsletters_limits = self._get_column_limits("newsletters")
        investors_limits = self._get_column_limits("investors")

        print(f"📊 Found {len(emails)} emails to process")
        total_funding_rounds = 0

        for i, email in enumerate(emails, 1):
            clean_subj = self.clean_subject(email.get('subject', '')).strip()
            print(f"\n📧 Processing email {i}/{len(emails)}")
            print(f"   Subject: {clean_subj[:50]}...")

            try:
                email_date = date_parser.parse(email['date']).date()
            except:
                email_date = datetime.now().date()

            try:
                subject_trunc = clean_subj[:newsletters_limits.get('subject', 500)] if newsletters_limits.get('subject') else clean_subj
                newsletter_id = self.insert_newsletter(subject_trunc, email_date,
                                                       message_id=email.get('message_id'),
                                                       processed=False)
            except Exception as e:
                print(f"❌ Failed to insert newsletter for email {i}: {e}")
                continue

            funding_rounds = self.parse_funding_rounds(email['body'], email_date)
            print(f"   Found {len(funding_rounds)} funding rounds")

            for funding in funding_rounds:
                try:
                    funding['newsletter_id'] = newsletter_id
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

                    company_id = self.insert_company(funding)
                    funding_id = self.insert_funding_round(company_id, funding)

                    for inv_name in funding.get('investors', []):
                        try:
                            inv_trunc = inv_name[:investors_limits.get('name', 255)] if investors_limits.get('name') else inv_name
                            inv_id = self.insert_investor(inv_trunc)
                            self.link_funding_investor(funding_id, inv_id)
                        except Exception as e:
                            print(f"      ❌ Investor insert/link error '{inv_name}': {e}")

                    self.conn.commit()
                    print(f"   ✅ {funding['company_name']} - €{funding['amount_eur']:,.0f}")
                    total_funding_rounds += 1

                except Exception as e:
                    self.conn.rollback()
                    print(f"   ❌ Error inserting funding '{funding.get('company_name','Unknown')}': {e}")
                    continue

        print(f"\n🎉 Import complete! Added {total_funding_rounds} funding rounds with investors/newsletters links")


def main():
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'EU_Startups',
        'user': 'postgres',
        'password': 'Shubham2100'
    }

    parser = StartupFundingParser(db_config)
    if not parser.connect_to_db():
        return

    try:
        parser.process_emails_json('emails.json')
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    finally:
        parser.close_connection()


if __name__ == "__main__":
    main()
