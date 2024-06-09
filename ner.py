from gliner import GLiNER

from schema import Entity, Iterable


PRETRAINED_CHECKPOINT = 'urchade/gliner_multi-v2.1'  # multi-lingual
# PRETRAINED_CHECKPOINT = 'numind/NuNerZero'  # English only
REVISION = '853ce23e47e519248ba3ec5953f002a80bffdedd'  # for GLiNER multi
DEFAULT_LABELS = ('person', 'organisation')


class EntityFinder:
    """
    EntityFinder finds entity in texts given a set of labels for which to look
    """
    def __init__(self, labels: Iterable[str] = DEFAULT_LABELS, pretrained_checkpoint=PRETRAINED_CHECKPOINT, revision=REVISION):
        # NOTE: NuZero requires labels to be lower-cased!
        self.labels = [label.lower() for label in labels]
        self.model = GLiNER.from_pretrained(pretrained_checkpoint, revision=revision)
    
    def find_entities_in_texts(self, *texts: str, threshold: int = 0.5):
        for text in texts:
            new_entities = self.model.predict_entities(text, self.labels)
            new_entities = merge_entities(new_entities)
            new_entities = (
                Entity(name=entity['text'], label=entity['label'])
                for entity in new_entities
                if entity['score'] > threshold
            )
            yield from new_entities


def merge_entities(entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
            current['text'] = text[current['start']: next_entity['end']].strip()
            current['end'] = next_entity['end']
        else:
            merged.append(current)
            current = next_entity
    # Append the last entity
    merged.append(current)
    return merged

