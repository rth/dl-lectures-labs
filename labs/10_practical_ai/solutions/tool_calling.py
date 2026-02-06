import json

# Define the tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": (
                "Search a product database by query and optional category filter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "electronics",
                            "books",
                            "clothing",
                            "food",
                            "toys",
                        ],
                        "description": "Optional category to filter results",
                    },
                },
                "required": ["query"],
            },
        },
    }
]


# Mock database function
def search_database(query, category=None):
    mock_db = {
        "electronics": [
            {"name": "Wireless Headphones", "price": 79.99, "rating": 4.5},
            {"name": "USB-C Hub", "price": 34.99, "rating": 4.2},
            {"name": "Mechanical Keyboard", "price": 129.99, "rating": 4.8},
        ],
        "books": [
            {"name": "Deep Learning with Python", "price": 49.99, "rating": 4.7},
            {"name": "AI: A Modern Approach", "price": 89.99, "rating": 4.6},
        ],
        "clothing": [
            {"name": "Running Shoes", "price": 119.99, "rating": 4.3},
        ],
    }
    results = []
    for cat, items in mock_db.items():
        if category and cat != category:
            continue
        for item in items:
            if query.lower() in item["name"].lower():
                results.append({**item, "category": cat})
    if not results:
        # Return some default results if no exact match
        all_items = [
            {**item, "category": cat}
            for cat, items in mock_db.items()
            for item in items
            if not category or cat == category
        ]
        results = all_items[:3]
    return results


# Step 1: Send user message with tools
user_message = "Find me some good electronics for a developer"
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful shopping assistant. Use the search_database"
            " tool to find products."
        ),
    },
    {"role": "user", "content": user_message},
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
)

# Step 2: Check if the model wants to call a tool
assistant_message = response.choices[0].message
print("Assistant response:", assistant_message)

if assistant_message.tool_calls:
    # Step 3: Execute the tool call
    tool_call = assistant_message.tool_calls[0]
    function_args = json.loads(tool_call.function.arguments)
    print(f"\nTool call: {tool_call.function.name}({function_args})")

    result = search_database(**function_args)
    print(f"Tool result: {json.dumps(result, indent=2)}")

    # Step 4: Send tool result back to model
    messages.append(assistant_message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        }
    )

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    print(f"\nFinal answer: {final_response.choices[0].message.content}")
else:
    print(f"No tool call. Response: {assistant_message.content}")
