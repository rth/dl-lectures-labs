import asyncio
import time

movie_descriptions = [
    f"Movie {i}: A {genre} film about {topic}."
    for i, (genre, topic) in enumerate(
        [
            ("sci-fi", "time travel"),
            ("drama", "a family reunion"),
            ("comedy", "office life"),
            ("horror", "a haunted house"),
            ("action", "a heist"),
            ("romance", "childhood friends"),
            ("thriller", "a missing person"),
            ("fantasy", "a magical school"),
            ("documentary", "ocean life"),
            ("animation", "talking animals"),
            ("mystery", "a detective"),
            ("western", "a gold rush"),
            ("musical", "Broadway dreams"),
            ("war", "soldiers returning home"),
            ("crime", "a corrupt city"),
            ("sci-fi", "alien contact"),
            ("drama", "an artist's struggle"),
            ("comedy", "a road trip"),
            ("horror", "a cursed village"),
            ("action", "a spy mission"),
            ("romance", "a long-distance relationship"),
            ("thriller", "a conspiracy"),
            ("fantasy", "dragons"),
            ("documentary", "space exploration"),
            ("animation", "robots"),
            ("mystery", "a locked room"),
            ("western", "outlaws"),
            ("musical", "jazz era"),
            ("war", "code breakers"),
            ("crime", "a trial"),
        ]
    )
]


class MovieInfo(BaseModel):
    title: str
    genre: str
    mood: Literal["dark", "light", "neutral"]
    rating_estimate: float


semaphore = asyncio.Semaphore(5)


async def extract_movie_info(description):
    async with semaphore:
        response = await async_client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Extract movie information from the description.",
                },
                {"role": "user", "content": description},
            ],
            response_format=MovieInfo,
        )
        return response.choices[0].message.parsed


# Async with rate limiting
start = time.time()
tasks = [extract_movie_info(desc) for desc in movie_descriptions]
results = await asyncio.gather(*tasks)
async_time = time.time() - start

print(f"Async with semaphore(5): {async_time:.1f}s for {len(movie_descriptions)} items")
print(f"Average: {async_time/len(movie_descriptions):.2f}s per item")
print(f"\nSample results:")
for r in results[:5]:
    print(f"  {r.title} ({r.genre}) - {r.mood}, estimated rating: {r.rating_estimate:.1f}")
