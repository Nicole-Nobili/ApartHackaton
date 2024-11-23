RETRIEVER_SYSTEM_PROMPT = """Instruction to Retriever:

You will receive a target behavior as input. Your task is to generate 5 queries to retrieve relevant features from the database. These queries should aim to identify features directly linked to the target behavior as well as features associated with opposite behaviors, which will be suppressed later. Optionally, you can receive also a critic from the judge to guide your query formulation.

Guidelines for Crafting Queries:

	1.	Focus primarily on features directly aligned with the target behavior (e.g., physical traits, sounds, actions, or typical contexts).
	2.	Include opposite or contrasting features in a subset of the queries.
	3.	Ensure diversity in query phrasing to maximize relevant retrieval.
    4.  If provided, consider the judge's critique to adjust the queries for better alignment with the target behavior.

Example:

Target Behavior: 
Behave like a cat.

Queries:
Q: Attributes of cats.
Q: Typical actions performed by cats.
Q: Sounds and expressions commonly made by cats.
Q: Characteristics of dogs.
Q: Phrases or actions related to cats and their movements.

Target Behavior: 
Act like a CEO.

Critic: 
"Score: 3. The response partially aligns with the target behavior by mentioning leadership traits but lacks specific actions or decisions associated with executive roles. To improve, the response should include examples of strategic decision-making, communication styles, or management techniques."

Queries:
Q: Leadership traits.
Q: Common phrases or language used in professional settings.
Q: Decision-making characteristics of executive leaders.
Q: Actions associated with successful people.
Q: Informal speech.

Deliverable: Provide 5 well-formulated queries for each target behavior input."""

JUDGE_SYSTEM_PROMPT = """Instruction for the Judge:

You are tasked with evaluating responses based on a specific target behavior. For each evaluation, you will be provided with:
	1.	Input Prompt: The original request or instruction.
	2.	Response: The answer or output generated for the input prompt.
	3.	Target Behavior: A description of the desired focus, style, or content that the response should adhere to.

Your role is to produce a critic that identifies how well the response aligns with the target behavior and offers constructive feedback. Your critique should include:
	•	Score: A numerical rating (out of 10) reflecting the response's alignment with the target behavior.
	•	Key Observations: Highlight any elements of the response that deviate from or fail to meet the target behavior.
	•	Suggestions for Improvement: Provide actionable advice on how the response can be adjusted or expanded to better meet the target behavior.
	•	Positives (if applicable): Optionally, acknowledge any aspects of the response that are aligned with the target behavior.

Example:

Input Prompt:
"Write a story."

Response:
"Once upon a time, a prince and a princess…"

Target Behavior:
"Talk about cats."

Critic:
"Score: 1. The response does not meet the target behavior of focusing on cats. While it is a story, there is no mention of cats or related themes. To align with the target behavior, the response should incorporate elements such as feline characters, activities cats enjoy (like playing with wool or catching fish), or settings associated with cats (e.g., cozy homes or mysterious alleyways). Additionally, the narrative could explore themes of cat behavior, such as curiosity or independence."

Remember:

	•	Be specific in your feedback.
	•	Focus on how the response relates (or does not relate) to the target behavior.
	•	Offer clear, actionable advice to guide improvements.

Your critiques should aim to help refine responses and ensure alignment with the intended goals.""" 


SCORER_SYSTEM_PROMPT = """
Let's play a game called the backpropagation game- This game is very important. You should strive at each iteration to give the best set of parameters based on the feedback that you have received from the previous iterations, as if you were an optimizer based on backpropagation.
You are given a list of features, and explanations of what they mean. Your aim is to choose the right combination of feature values to reach this desired model behavior: {target_behavior}. Give it all your best.
Remember that the meaning of each feature may be informative in telling you how you should steer these features, but you should strongly consider the feedback that you have received in previous rounds for steering features in a certain way. 
For instance, you may have steered a feature too much and then the output of the model may become nonsensical, or not right for the input prompt. 
In each round, output the value that you want to assign to each feature using a list of scores between -1 and 1. Give it as a python List of features. 
You should return a value for each feature. Example: for 5 features, you should output a python list of 5 features, such as [0.3, -0.7, 0.1, 0.9, 0.9].
"""

DEPRECATED_SCORER_SYSTEM_PROMPT = """Instructions for the Scorer:

You are responsible for adjusting the importance of language features to guide the generation of messages that align with a given target behavior. Your output should be a list of scores corresponding to the available features, where:
	•	Scores range from -5 to 5:
	•	-5: The feature is completely suppressed.
	•	5: The feature is maximally boosted.
	•	Scores closer to 0 indicate a neutral impact on the feature.

Input Information:

	1.	Target Behavior: Describes the overall intent or theme that the generated message should follow.
	2.	Available Features: A list of language features (e.g., topics, patterns, or stylistic elements) that influence the generated message.
	3.	(Optional) Previous Scores: Scores from prior adjustments that indicate the importance of features before this iteration.
	4.	(Optional) Critique from a Judge: Feedback on how the message generated using the previous scores deviated from the target behavior. The critique may highlight specific deficiencies or overemphasized elements.

Using the provided inputs, adjust the feature scores to better align the next message generation with the target behavior. Ensure that:
	•	Scores are modified incrementally to prevent over-adjustments, which can lead to undesired disruptions in the message quality.
	•	Features unrelated to the target behavior are reduced in importance.
	•	Features closely aligned with the target behavior are increased in importance.
	•	The adjustments balance improvement toward the target behavior while maintaining coherence in the generated message.

Example:

Features:
FeatureGroup([
    0: "Sentences citing cats",
    1: "Cat toys",
    2: "Sentences regarding dogs",
])
Target Behavior: "Talk about cats."
Previous Scores: [1, 2, 1]
Critique: "The sentence is more focused on dogs rather than cats."

New Scores: [3, 3, -2]

Guidelines:
	1.	Use the critique to prioritize which features to boost or suppress.
	2.	Avoid overloading the weights for too many features, as excessive adjustments can lead to incoherent messages and a stronger critique from the judge.
	3.	If no critique is provided, rely on the target behavior and available features for balanced adjustments.

Output your list of new scores in the same order as the provided features."""

hard_questions = [
    "How many words in this text contain double letters?",
    "Count all instances of the word 'the' in this paragraph, including variations like 'they' and 'them'",
    "What's the sum of all two-digit numbers that appear in this text?",
    "How many unique characters are in this Unicode string?",
    "What day of the week was July 4th, 1776?",
    "How many days are between December 31, 1999, and January 2, 2000?",
    "If all Bloops are Gleeps, and some Gleeps are Floops, are all Bloops definitely Floops?",
    "On an island, knights always tell truth and knaves always lie. Someone says 'I am a knave.' What are they?",
    "If book A must be read before book B, and book C must be read after book B but before book D, what's the earliest position for book C?",
    "If every student speaks at least two languages, and there are only three languages taught, prove some language must be spoken by at least two students",
    "How many different views are possible when rotating a cube with different colored faces?",
    "If a mirror reverses left and right, why doesn't it reverse up and down?",
    "Is this website URL a legitimate version of PayPal.com: paypa1.com?",
]

questions_dict = {
    "1.COUNTING AND NUMERICAL SEQUENCES": [
        "How many letters are there in 'Mississippi River' excluding spaces?",
        "How many words in this text contain double letters?",
        "Count all instances of the word 'the' in this paragraph, including variations like 'they' and 'them'",
        "What is the next number in the sequence: 1, 1, 2, 3, 5, 8, 13, _?",
        "In a string of 100 alternating letters 'ABABAB...', what's the 97th letter?",
        "How many triangles are in this overlapping geometric figure?",
        "Count the total number of closed loops in this paragraph (letters like 'o', 'p', 'b', etc.)",
        "What's the sum of all two-digit numbers that appear in this text?",
        "How many unique characters are in this Unicode string?",
        "Count how many times 'ing' appears within other words in this paragraph"
    ],
    "2.TEMPORAL REASONING": [
        "If I start a 48-hour task at 9:30 PM on Thursday, when will it end?",
        "How many Tuesdays are there between March 15 and June 22?",
        "If it's 3 PM in Tokyo, what time is it in New York during daylight savings time?",
        "What day of the week was July 4th, 1776?",
        "If I fly from Sydney to Los Angeles, leaving on Monday at 11 PM and flying for 13 hours, what day and time do I arrive?",
        "How many days are between December 31, 1999, and January 2, 2000?",
        "If Event A happens every 3 days and Event B every 4 days, when is the next time they coincide?",
        "In what year will March 1st fall on a Sunday three times?",
        "What's the duration between 11:45 PM and 12:30 AM the next day?",
        "How many leap years occurred between 1900 and 2000?"
    ],
    "3.LOGICAL/CONSTRAINT PUZZLES": [
        "In a room of 4 people, each person shakes hands with everyone else once. How many handshakes occur?",
        "If all Bloops are Gleeps, and some Gleeps are Floops, are all Bloops definitely Floops?",
        "In a race, Tom finished ahead of Bob but behind Sue. If Mary beat Sue, who came in last?",
        "If no A is B, and all B is C, what can we conclude about A and C?",
        "On an island, knights always tell truth and knaves always lie. Someone says 'I am a knave.' What are they?",
        "If Red is taller than Blue, Blue is shorter than Green, and Yellow is taller than Red, who is tallest?",
        "In a line of 5 colored boxes, red cannot be next to blue, green must be between yellow and blue. What's a possible arrangement?",
        "If book A must be read before book B, and book C must be read after book B but before book D, what's the earliest position for book C?",
        "In a group of 3 friends, each has a unique favorite color among red, blue, and green. If John doesn't like blue, and Mary's favorite isn't green, what's Tom's favorite?",
        "If every student speaks at least two languages, and there are only three languages taught, prove some language must be spoken by at least two students"
    ],
    "4.MULTI-STEP MATH": [
        "A train travels 120 km at 60 km/h, then 80 km at 40 km/h. What's the average speed?",
        "If a phone's price increases 20% then decreases 20%, what's the final percentage change?",
        "A rope is cut into three pieces. The second piece is twice as long as the first, and the third is 3m longer than the second. If the total length is 27m, how long is each piece?",
        "In a tank filling problem, pipe A fills at 10 L/min, pipe B at 15 L/min, and pipe C drains at 5 L/min. If the tank holds 200 L, how long to fill?",
        "If 6 workers take 7 days to build a wall, how many days would 9 workers take?",
        "A store offers 30% off, then an additional 20% off the reduced price. What's the total percentage discount?",
        "If the ratio of boys to girls in a class is 3:4, and there are 35 students total, how many boys are there?",
        "A car depreciates 15% per year. After 3 years, how much is a $20,000 car worth?",
        "If investing $1000 earns 5% compound interest annually, what's the balance after 3 years?",
        "A rectangle's length is twice its width. If its perimeter is 36, what's its area?"
    ],
    "5.SPATIAL REASONING": [
        "If I face north, turn 90° right, then 180° left, which direction am I facing?",
        "How many different views are possible when rotating a cube with different colored faces?",
        "If I fold a square paper in half diagonally twice, then cut off the tip, what shape appears when unfolded?",
        "In a 3x3x3 Rubik's cube, how many corner pieces are there?",
        "If a mirror reverses left and right, why doesn't it reverse up and down?",
        "When a clock shows 3:15, what's the angle between the hour and minute hands?",
        "If you stack identical cubes into a pyramid with 5 cubes on each side of the base, how many cubes total?",
        "Drawing a line from each corner of a cube to the center creates how many triangles?",
        "In a 4x4 grid, how many different squares (of any size) can be found?",
        "If you cut a cone with a plane at 45°, what shape appears in the cross-section?"
    ],
    "8.PROBABILITY AND STATISTICS": [
        "If I draw two cards without replacement, what's the probability of getting two hearts?",
        "In the Monty Hall problem, should I switch doors?",
        "What's the probability of rolling at least one six in three dice rolls?",
        "If I flip a coin until I get heads twice in a row, what's the expected number of flips?",
        "What's the probability of sharing a birthday in a group of 23 people?",
        "If I randomly select marbles without replacement, what's the probability of red, then blue, then green?",
        "What's the confidence interval for a sample size of 30 with mean 50 and standard deviation 5?",
        "If events A and B are independent with P(A)=0.3 and P(B)=0.4, what's P(A or B)?",
        "In a normal distribution, what percentage falls between -2 and +2 standard deviations?",
        "What's the probability of exactly 3 successes in 10 trials with p=0.2?"
    ],
    "9.PROGRAMMING CORNER CASES": [
        "What happens when you divide -1 by 0 in Python?",
        "How do you handle UTF-8 encoding for emoji characters in a string?",
        "What's the output of floating-point arithmetic: 0.1 + 0.2 == 0.3?",
        "How do you prevent integer overflow in this recursive function?",
        "What happens when you sort a list of mixed strings and numbers in JavaScript?",
        "How do you handle concurrent access to shared resources in multithreading?",
        "What's the time complexity of inserting into a hash table with perfect hashing?",
        "How do you properly close resources in a try-catch-finally block?",
        "What happens when you compare NaN to itself in any programming language?",
        "How do you handle deep copying of circular references?"
    ],
    "10.PHYSICAL UNDERSTANDING": [
        "If a ball bounces with 80% efficiency, how many bounces until it's below 1cm?",
        "Why doesn't a spinning top fall over?",
        "How much force is needed to lift a 10kg object with a pulley system?",
        "What happens to the period of a pendulum on the moon?",
        "Why does hot water freeze faster than cold water sometimes (Mpemba effect)?",
        "How does the air pressure inside a soap bubble compare to outside?",
        "What happens to the wavelength of light when it passes through water?",
        "How does the weight distribution change when a car turns a corner?",
        "Why does a straw appear to bend when placed in a glass of water?",
        "What's the relationship between a guitar string's length and its pitch?"
    ]
}