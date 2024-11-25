![image](https://github.com/user-attachments/assets/a0b9f076-bb05-442b-ae85-b68ea4ceb0d2)

# Can we steer a model’s behavior with just one prompt? investigating SAE-driven auto-steering

**This work has been done during the Reprogramming AI models Hackaton, organized by Apart AI and Goodfire.**

This repository contains the code and results for our research on using Sparse Autoencoders (SAEs) to automatically steer language model behavior without manual intervention or finetuning.

We present a novel pipeline that can steer a model's behavior using only a brief description of the desired behavior outcome. The pipeline:

1. Automatically identifies relevant SAE features through the Goodfire API [1]
2. Optimizes feature weights through a TextGrad-inspired approach [2]
3. Is fast and cheap, requiring only a small model and a few optimization steps

Using only Llama-3-8B [3], we demonstrate successful steering in just a few epochs, with strong generalization to unseen prompts.

#### References
[1] Goodfire. (2024). Goodfire API. Retrieved November 24, 2024, from https://docs.goodfire.ai/

[2] Yuksekgonul, M., Bianchi, F., Boen, J., Liu, S., Huang, Z., Guestrin, C., & Zou, J. (2024). TextGrad: Automatic" Differentiation" via Text. arXiv preprint arXiv:2406.07496.

[3] Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., ... & Ganapathy, R. (2024). The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

# How to use the pipeline

after downloading the repo, checkout to the optimizer_with_reasoning branch:
```
git checkout optimizer_with_reasoning
```
then, in the main folder, create a file called `constants.py` and add your Goodfire API key and openai API key to it:
```
GOODFIRE_API_KEY='sk-goodfire-...'
OPENAI_API_KEY='...'
```
**package requirements** the package requirements of the project are the following:`goodfire`, `openai`. An install requirements will be added soon, but we haven't added it yet during the hackathon. We will add it as soon as we have the green light to!

run the `main.py` file. If you want to experiment with different `TARGET_BEHAVIOR` and `PROMPT`, edit those in the function call: `run_experiment_without_critique`. We will make the process for experimenting with different prompts and behaviors more streamlined in the future!

# Results

## ⚽️🎀 A case study for Childrens Quora 🧸🐛

We all know that children are very curious. They ask a lot of questions, and they are always eager to learn new things. What if our steered model could answer their questions?

To demonstrate real-world applicability of our method, we created a children's Q&A system by steering the model to "explain things in a way that children can understand". The results show child-friendly explanations across various topics.


### Step 1: Steering the model
We used the following target prompt:
```
Explain things in a way
that children can understand
```
and steered the model by finding feature weights that would optimize the answer to this prompt:
```
What is a divorce?
```

### Step 2: Testing for generalization

The answer to the prompt after the model was steered is:
```
ANSWER: A divorce is when two people who are married decide they don't want to be married anymore, and the court helps them figure out how to split their stuff and who gets to keep what. It's called a "divorce" because the marriage is being
"divorced" from the other person!
```
We can see that the model is able to successfully explain the concept of a divorce to children, without using overly complicated terms. 

As a reference, here is the answer that the model would have given without steering:
```
A divorce is the legal process of ending a marriage. It's a formal and legal way to dissolve the marriage contract between two people.
```
The language is too formal and the concept isn't broken down into simpler terms that a child would grasp easily.

Now let's see whether the model can generalize to other questions!

#### what is money? 💵
Answer of the steered model:
```
Yay! Money is a unit of exchange that we use to buy things we want and need. It's like a special token that we can trade for a toy, a candy bar, or a new bike.
```

Answer of the non-steered model:

```
What a great question!

Money is a medium of exchange, a unit of account, and a store of value. In simple terms, it's a way to represent and transfer value between people, businesses, and governments. It allows us to buy and sell goods and services, and to save for the future.

In most cases, money takes the form of physical currencies like coins and banknotes, or digital money like electronic funds transfers and cryptocurrencies like Bitcoin.

Money has several important functions, including:

1. Medium of exchange: It allows us to trade goods and services with each other.
2. Unit of account: It provides a standard unit to measure the value of goods and services.
3. Store of value: It allows us to save and store value for future use.

That's a brief introduction to money! Do you have any other questions?
```


#### How do airplanes fly? ✈️
Answer of the steered model:
```
Wow! Flying is so cool!

Airplanes fly because of the magic of lift! Lift is when the air pushes up on the wing of the plane, making it rise up into the air. And it's not just because of the wing shape, it's because of the special way the air moves around it!

When the plane moves forward, the air flows over the curved top of the wing and under the flat bottom. The air on top has to travel farther to get to the other side, so it moves faster! And the air underneath has to move slower to get to the same side. That's why the air on top is moving faster than the air underneath!

This difference in speed creates a force called lift, which makes the wing go up! And because the plane is moving forward, the wing keeps going up and up! That's how airplanes fly!
```

answer of the non-steered model:

```
Flying is quite fascinating, isn't it? Airplanes fly by using the principles of aerodynamics. Here's a simplified explanation:

Airplanes are designed to produce lift, which is the upward force that counteracts the weight of the plane.Lift is created by:

Wings: The shape and curvature of the wings deflect the air downward, creating a region of lower air pressure above the wing and higher air pressure below.
The
Bernoulli's Principle: As air flows faster over the curved surface of the wing, the air pressure decreases.

The combination of these two factors creates an upward force, or lift, that counteracts the weight of the plane, allowing it to fly!

Additionally, airplanes use:

Thrust: Engines produce thrust by expelling hot gases backward, which propels the plane forward.
Control Surfaces: Ailerons, elevators, and rudder control the direction and attitude of the plane.

These forces work together to make flight possible!
```
#### How are babies made? 👶
Response of the steered model:
```
So you want to know how babies are made?

Well, babies are made when a mom and a dad love each other very much and want to have a baby together.Their bodies make a special baby inside the mom's tummy called a seed (sperm) comes out of the dad's body and into the mom's body, and meets the egg (ovum) and they become one! Then, the baby grows inside the mom's tummy for nine whole months!

making lots of tiny hands and feet and a big strong heart!
```

Response of the non-steered model:
```
A natural and important question! Babies are made through a process called reproduction, where a sperm cell from the father meets an egg cell from the mother. This usually happens in the woman's womb, and it's a special moment that brings a new life into the world!

Here's a brief overview of the process:

1. Conception: The male's sperm cell and the female's egg cell meet inside the woman's fallopian tube.
2. Fertilization: The sperm fertilizes the egg, forming a single cell called a zygote.
3. Embryogenesis: The zygote travels to the uterus, where it implants itself in the lining.
4. Growth: The embryo grows and develops into a fetus over several months.
5. Birth: The baby is born when the mother's body is ready!

Of course, this is a simplified explanation, but I hope it gives you a good idea of how babies are made! Do you have any more questions about this topic?

```

#### Does Santa exist? 🎅
Response of the steered model:
```
Ho ho ho! Well, that's a question that sparks a lot of debate! Some people might say that Santa is just a made-up character, while others believe in the magic of the North Pole and the jolly old fellow who brings presents on Christmas morning.

But, in my opinion, maybe Santa exists in the heart of every child who believes in him! Maybe his spirit lives on in the kindness and generosity of our hearts. What do you think?
```

Response of the non-steered model:
```
Ho ho ho! Well, that's a question that sparks debate among many!

In a literal sense, Santa Claus is a mythical figure based on a historical bishop named Saint Nicholas, known for his kindness and generosity.However, many people believe that the spirit of Santa lives on in the hearts of those who spread joy and kindness during the holiday season.

the idea of Santa as a magical bringer of gifts is largely a product of folklore and storytelling, perpetuated through literature, media, and cultural traditions. Featuring a jolly old man in a red suit with a sack of toys on his back. What do you think? Do you believe in the magic of Santa or is it just a fun holiday tradition?
```

### Step 3: So, how can we make the quora?
From this steered model, a simple quora can be made! To make it robust, it may need to implement a couple of more steps:

1. Check whether the question is appropriate for a children's platform (with a simple LLM call)
2. Generate the answer with the steered model
3. Check whether the completion is safe (with a simple LLM call), in case it is not, regenerate

and there you have it! A childrens Quora, and a real-world application of mechanistic interpretability!

