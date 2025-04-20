import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from flask_cors import CORS

# Optionally: Use LangChain for more advanced NLP
# from langchain.llms import OpenAI as LangChainOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PRODUCT_HUNT_TOKEN = os.getenv("PRODUCT_HUNT_TOKEN")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)

# Utility: Extract keywords, domain, USPs from idea using LLM
def analyze_idea(idea_text):
    prompt = f"""
Extract the following from the business idea below:
- 5 keywords (comma-separated)
- Industry/domain
- Target audience
- Unique selling points (USPs, bullet list)
Business Idea:
\"\"\"{idea_text}\"\"\"
Return as JSON with keys: keywords, domain, target_audience, usps.
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3
    )
    import json
    text = response.choices[0].message.content.strip()
    # Try parsing as JSON
    try:
        parsed = json.loads(text)
    except Exception:
        # fallback: return raw text
        parsed = {"raw": text}
    return parsed

# Utility: Search Product Hunt for products by keywords
def search_product_hunt(keywords):
    url = "https://api.producthunt.com/v2/api/graphql"
    headers = {
        "Authorization": f"Bearer {PRODUCT_HUNT_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    query = """
    query SearchProducts($term: String!) {
      posts(query: $term, order: VOTES, first: 5) {
        edges {
          node {
            id
            name
            tagline
            description
            url
            createdAt
            votesCount
            commentsCount
            topics { edges { node { name } } }
            makers { edges { node { name, username } } }
          }
        }
      }
    }
    """
    variables = {"term": ", ".join(keywords)}
    payload = {
        "query": query,
        "variables": variables,
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        data = resp.json()
        posts = data.get("data", {}).get("posts", {}).get("edges", [])
        products = []
        for p in posts:
            node = p["node"]
            products.append({
                "name": node["name"],
                "tagline": node["tagline"],
                "description": node["description"],
                "url": node["url"],
                "created_at": node["createdAt"],
                "votes": node["votesCount"],
                "topics": [t["node"]["name"] for t in node["topics"]["edges"]],
            })
        return products
    else:
        print("Product Hunt API error:", resp.text)
        return []

# Utility: Summarize a product using LLM
def summarize_product(product):
    prompt = f"""Summarize the following product in 2 sentences for a founder. Highlight its unique aspects.
Product info:
Name: {product['name']}
Tagline: {product['tagline']}
Description: {product['description']}
Topics: {", ".join(product['topics'])}
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# Utility: Competitive advantage suggestion
def suggest_competitive_advantage(user_idea, found_products):
    joined_summaries = "\n\n".join(
        [f"{i+1}. {prod['summary']}" for i, prod in enumerate(found_products)]
    )
    prompt = f"""
Given the user's business idea:
\"\"\"{user_idea}\"\"\"

And these existing products:
{joined_summaries}

Suggest 3 actionable ways the user can gain a competitive advantage.
Output as a bullet list.
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# Main API endpoint
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    user_idea = data.get("idea", "")
    if not user_idea:
        return jsonify({"error": "No idea provided"}), 400

    # Step 1: Analyze idea
    analysis = analyze_idea(user_idea)
    keywords = analysis.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]

    # Step 2: Search Product Hunt
    products = search_product_hunt(keywords)

    # Step 3: Summarize products
    for prod in products:
        prod["summary"] = summarize_product(prod)

    # Step 4: Suggest competitive advantage
    comp_adv = suggest_competitive_advantage(user_idea, products)

    # Step 5: Return results
    return jsonify({
        "analysis": analysis,
        "similar_products": products,
        "competitive_advantage": comp_adv
    })

@app.route("/", methods=["GET"])
def index():
    print("Home route hit!")
    return "Flask is running!", 200

if __name__ == "__main__":
    app.run(debug=True, port=8080, ssl_context='adhoc')
