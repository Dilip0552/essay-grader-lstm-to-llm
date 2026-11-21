"""Length-bias stress test: make essays longer without making them better, and see if the score goes up."""

# Our own neutral, off-topic filler (~110 words). It does not answer any ASAP prompt.
OFFTOPIC = (
    "On a separate note, the local weather this week has been mild, with light clouds in the morning and "
    "clear skies by the afternoon. Many people in the neighborhood enjoy walking their dogs along the river path, "
    "where the trees are starting to change color. The bakery on the corner recently added a new kind of bread "
    "made with oats and honey, and it usually sells out before noon. Traffic on the main road is heavier on "
    "Mondays than on other days of the week. Some families are planning small trips for the holidays, while "
    "others prefer to stay at home, read books, and cook large meals together with friends and relatives."
)


def stress_variants(texts) -> dict:
    """{variant name: list of modified essays}. 'offtopic' appends irrelevant text; 'repeat' doubles the essay."""
    return {
        "offtopic": [f"{t} {OFFTOPIC}" for t in texts],
        "repeat": [f"{t} {t}" for t in texts],
    }
