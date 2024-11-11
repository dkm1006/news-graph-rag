from relik import Relik
from relik.inference.data.objects import RelikOutput, TaskType, Document

from news_graph_rag.config import RELATION_EXTRACTION_MODEL
from news_graph_rag.schema import Relation


ALLOWED_RELATIONS = tuple(
    Document(**data) for data in (
    {
        "text": "member of",
        "id": 1,
        "metadata": {
            "description": "organization, club or musical group to which the subject belongs. Do not use for membership in ethnic or social groups, nor for holding a political position, such as a member of parliament (use P39 for that)",
            "property": "P463"
        }
    },
    {
        "text": "part of",
        "id": 2,
        "metadata": {
            "description": "object of which the subject is a part (if this subject is already part of object A which is a part of object B, then please only make the subject part of object A), inverse property of \"has part\" (P527, see also \"has parts of the class\" (P2670))",
            "property": "P361"
        }
    },
    {
        "text": "located in",
        "id": 3,
        "metadata": {
            "description": "location of the object, structure or event. In the case of an administrative entity as containing item use P131. For statistical entities use P8138. In the case of a geographic entity use P706. Use P7153 for locations associated with the object",
            "property": "P276"
        }
    },
    {
        "text": "has position",
        "id": 4,
        "metadata": {
            "description": "subject currently or formerly holds the object position or public office",
            "property": "P39"
        }
    }
    )
)


reader = Relik.from_pretrained(RELATION_EXTRACTION_MODEL, retriever=None)
# entity_mentions = entity_finder.find(text)
# start_end_tuples = [(e.start, e.end) for e in entity_mentions]


def extract_relations(texts: list[str], allowed_relations=ALLOWED_RELATIONS):
    relik_out: list[RelikOutput] = reader(
        texts,
        candidates={TaskType.TRIPLET: [allowed_relations for _ in texts]},
    )
    triplets = [
        Relation(triplet.subject, triplet.object, to_graph_label(triplet.label))
        for output in relik_out
        for triplet in output.triplets
    ]
    return triplets


def to_graph_label(relation_label: str) -> str:
    """Converts lower case relation label to UPPER_CASE"""
    return relation_label.replace(' ', '_').upper()
