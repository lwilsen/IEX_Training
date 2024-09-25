import streamlit as st

st.title("Comparison of sentiment evaluation of an LLM and NLP model")

reviews = [
    'In 1974, the teenager Martha Moxley (Maggie Grace) moves to the high-class area of Belle Haven, Greenwich, Connecticut. On the Mischief Night, eve of Halloween, she was murdered in the backyard of her house and her murder remained unsolved. Twenty-two years later, the writer Mark Fuhrman (Christopher Meloni), who is a former LA detective that has fallen in disgrace for perjury in O.J. Simpson trial and moved to Idaho, decides to investigate the case with his partner Stephen Weeks (Andrew Mitchell) with the purpose of writing a book. The locals squirm and do not welcome them, but with the support of the retired detective Steve Carroll (Robert Forster) that was in charge of the investigation in the 70\'s, they discover the criminal and a net of power and money to cover the murder.<br /><br />"Murder in Greenwich" is a good TV movie, with the true story of a murder of a fifteen years old girl that was committed by a wealthy teenager whose mother was a Kennedy. The powerful and rich family used their influence to cover the murder for more than twenty years. However, a snoopy detective and convicted perjurer in disgrace was able to disclose how the hideous crime was committed. The screenplay shows the investigation of Mark and the last days of Martha in parallel, but there is a lack of the emotion in the dramatization. My vote is seven.<br /><br />Title (Brazil): Not Available',
    "OK... so... I really like Kris Kristofferson and his usual easy going delivery of lines in his movies. Age has helped him with his soft spoken low energy style and he will steal a scene effortlessly. But, Disappearance is his misstep. Holy Moly, this was a bad movie! <br /><br />I must give kudos to the cinematography and and the actors, including Kris, for trying their darndest to make sense from this goofy, confusing story! None of it made sense and Kris probably didn't understand it either and he was just going through the motions hoping someone would come up to him and tell him what it was all about! <br /><br />I don't care that everyone on this movie was doing out of love for the project, or some such nonsense... I've seen low budget movies that had a plot for goodness sake! This had none, zilcho, nada, zippo, empty of reason... a complete waste of good talent, scenery and celluloid! <br /><br />I rented this piece of garbage for a buck, and I want my money back! I want my 2 hours back I invested on this Grade F waste of my time! Don't watch this movie, or waste 1 minute of your valuable time while passing through a room where it's playing or even open up the case that is holding the DVD! Believe me, you'll thank me for the advice!",
    '***SPOILER*** Do not read this, if you think about watching that movie, although it would be a waste of time. (By the way: The plot is so predictable that it does not make any difference if you read this or not anyway)<br /><br />If you are wondering whether to see "Coyote Ugly" or not: don\'t! It\'s not worth either the money for the ticket or the VHS / DVD. A typical "Chick-Feel-Good-Flick", one could say. The plot itself is as shallow as it can be, a ridiculous and uncritical version of the American Dream. The young good-looking girl from a small town becoming a big success in New York. The few desperate attempts of giving the movie any depth fail, such as the "tragic" accident of the father, the "difficulties" of Violet\'s relationship with her boyfriend, and so on. McNally (Director) tries to arouse the audience\'s pity and sadness put does not have any chance to succeed in this attempt due to the bad script and the shallow acting. Especially Piper Perabo completely fails in convincing one of "Jersey\'s" fear of singing in front of an audience. The only good (and quite funny thing) about "Coyote Ugly" is John Goodman, who represents the small ray of hope of this movie.<br /><br />I was very astonished, that Jerry Bruckheimer produced this movie. First "Gone In 60 Seconds" and now this... what happened to great movies like "The Rock" and "Con Air"? THAT was true Bruckheimer stuff.<br /><br />If you are looking for a superficial movie with good looking women just to have a relaxed evening, you should better go and see "Charlie\'s Angels" (it\'s much more funny, entertaining and self-ironic) instead of this flick.<br /><br />Two thumbs down (3 out of 10).',
    "hi for all the people who have seen this wonderful movie im sure thet you would have liked it as much as i. i love the songs once you have seen the show you can sing along as though you are part of the show singing and dancing . dancing and singing. the song ONE is an all time fave musical song too and the strutters at the end with the mirror its so oh you have to watch this one",
    "I recently bought the DVD, forgetting just how much I hated the movie version of \"A Chorus Line.\" Every change the director Attenborough made to the story failed.<br /><br />By making the Director-Cassie relationship so prominent, the entire ensemble-premise of the musical sails out the window.<br /><br />Some of the musical numbers are sped up and rushed. The show's hit song gets the entire meaning shattered when it is given to Cassie's character.<br /><br />The overall staging is very self-conscious.<br /><br />The only reason I give it a 2, is because a few of the great numbers are still able to be enjoyed despite the film's attempt to squeeze every bit of joy and spontaneity out of it.",
]

llm_responses = [
    """the sentiment of the text is mostly positive. the writer seems to have enjoyed the movie, praising its portrayal of a true story and the investigation process. while they do mention a lack of emotion in the dramatization, they still give the film a solid rating of seven, indicating they found it to be decent overall.""",
    """the sentiment of the text is negative. the writer expresses disappointment and frustration with the movie "disappearance" and its confusing storyline. they criticize the lack of a coherent plot and consider it a waste of time and talent. the text also includes strong negative words and exclamations that indicate the writer's displeasure with the movie.""",
    """the sentiment of the text is negative. the writer expresses disappointment and criticism towards the movie "coyote ugly," describing it as a waste of time, with a predictable plot, shallow characters, and failed attempts at depth. the acting and script are also criticized, and the overall recommendation is to avoid watching it.""",
    """the sentiment of the text is positive. the writer expresses their enjoyment of the movie, particularly its songs and interactive nature, and recommends it to others.""",
    """the sentiment of the text is negative. the writer expresses disappointment and frustration with the film adaptation of "a chorus line," criticizing the director's changes to the story, the rushed musical numbers, and the self-conscious staging. while the writer mentions a few enjoyable elements, the overall tone of the review is negative.]""",
]

st.write(
    """
5 Examples:
"""
)

for i in range(len(reviews)):
    st.text_area(f"Review {i + 1}", reviews[i], height=100)

st.write(
    f"""
| Review | Model Prediction | LLama (LLM) Prediction | Actual Value |
|---|---|---|---|
| 1 | Positive | "mostly positive" | 1 (positive) |
| 2 | Negative | "negative" | 0 (negative) |
| 3 | Negative | "negative" | 0 (negative) |
| 4 | Positive | "positive" | 1 (positive) |
| 5 | Negative | "negative" | 0 (negative) |
---
As you can see, both the NLP and LLM models correctly predicted all 5 examples correctly. However, their accuracy decreased when applied to larger batches of data."""
)

st.write(
    """
| Model | #N Samples | Accuracy |
|---|---|---|
| **LLM** | 200 | 0.775 |
| **NLP** | 200 | 0.717 |
| **LLM** | 1000 | 0.855 |
| **NLP** | 1000 | 0.793 |
"""
)
