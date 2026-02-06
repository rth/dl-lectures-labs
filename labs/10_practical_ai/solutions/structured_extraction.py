from pydantic import BaseModel
from typing import List, Literal


class ArticleInfo(BaseModel):
    headline: str
    topic: Literal[
        "politics", "technology", "business", "science", "sports", "entertainment"
    ]
    entities: List[str]
    sentiment: Literal["positive", "negative", "neutral"]


articles = [
    "Apple announced record quarterly revenue of $124 billion, driven by strong"
    " iPhone sales in emerging markets. CEO Tim Cook praised the company's"
    " innovation pipeline.",
    "The Senate passed a bipartisan infrastructure bill allocating $550 billion"
    " for roads, bridges, and broadband expansion across rural America.",
    "Scientists at CERN discovered a new subatomic particle that could reshape"
    " our understanding of quantum physics. The finding was published in Nature.",
    "Manchester United secured a dramatic 3-2 victory over Liverpool in the"
    " Premier League, with a last-minute goal from their star striker.",
    "Netflix reported a loss of 200,000 subscribers in Q1, sending shares"
    " tumbling 35% in after-hours trading. Analysts blame increased competition.",
]

for i, article in enumerate(articles):
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract structured information from the following news"
                    " article."
                ),
            },
            {"role": "user", "content": article},
        ],
        response_format=ArticleInfo,
    )
    info = response.choices[0].message.parsed
    print(f"\nArticle {i+1}:")
    print(f"  Headline:  {info.headline}")
    print(f"  Topic:     {info.topic}")
    print(f"  Entities:  {info.entities}")
    print(f"  Sentiment: {info.sentiment}")
