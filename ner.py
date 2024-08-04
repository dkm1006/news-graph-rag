from gliner import GLiNER

from schema import Entity, Iterable


PRETRAINED_CHECKPOINT = 'urchade/gliner_multi-v2.1'  # multi-lingual
# PRETRAINED_CHECKPOINT = 'numind/NuNerZero'  # English only
REVISION = '853ce23e47e519248ba3ec5953f002a80bffdedd'  # for GLiNER multi
DEFAULT_LABELS = ('person', 'organization')


class EntityFinder:
    """
    EntityFinder finds entity in texts given a set of labels for which to look
    """
    def __init__(self, labels: Iterable[str] = DEFAULT_LABELS, pretrained_checkpoint=PRETRAINED_CHECKPOINT, revision=REVISION):
        # NOTE: NuZero requires labels to be lower-cased!
        self.labels = [label.lower() for label in labels]
        self.model = GLiNER.from_pretrained(pretrained_checkpoint, revision=revision)
    
    def find(self, *texts: str, threshold=0.5):
        return list(self.find_iter(*texts, threshold=threshold))

    def find_iter(self, *texts: str, threshold=0.5):
        for text in texts:
            new_entities = self.model.predict_entities(text, self.labels, threshold=threshold)
            new_entities = merge_entities(text, new_entities)
            new_entities = (
                Entity(name=entity['text'], label=entity['label'])
                for entity in new_entities
            )
            yield from new_entities


def merge_entities(text, entities):
    """Merges entity tokens that directly follow each other"""
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
            current['text'] = text[current['start']:next_entity['end']].strip()
            current['end'] = next_entity['end']
        else:
            merged.append(current)
            current = next_entity
    # Append the last entity
    merged.append(current)
    return merged


if __name__ == '__main__':
    from utils import split_into_combined_sentence_chunks
    text_1 = "At the annual technology summit, the keynote address was delivered by a senior member of the Association for Computing Machinery Special Interest Group on Algorithms and Computation Theory, which recently launched an expansive initiative titled 'Quantum Computing and Algorithmic Innovations: Shaping the Future of Technology'. This initiative explores the implications of quantum mechanics on next-generation computing and algorithm design and is part of a broader effort that includes the 'Global Computational Science Advancement Project'. The latter focuses on enhancing computational methodologies across scientific disciplines, aiming to set new benchmarks in computational efficiency and accuracy."
    labels_1 = ["organization", "initiative", "project"]
    
    
    text_2 = """
    Snowflake Cortex gives you instant access to industry-leading large language models (LLMs) trained by researchers at companies like Mistral, Reka, Meta, and Google, including Snowflake Arctic, an open enterprise-grade model developed by Snowflake.
    Since these LLMs are fully hosted and managed by Snowflake, using them requires no setup. Your data stays within Snowflake, giving you the performance, scalability, and governance you expect.
    Snowflake Cortex features are provided as SQL functions and are also available in Python. The available functions are summarized below.
        COMPLETE: Given a prompt, returns a response that completes the prompt. This function accepts either a single prompt or a conversation with multiple prompts and responses.
        EMBED_TEXT_768: Given a piece of text, returns a vector embedding of 768 dimensions that represents that text.
        EMBED_TEXT_1024: Given a piece of text, returns a vector embedding of 1024 dimensions that represents that text.
        EXTRACT_ANSWER: Given a question and unstructured data, returns the answer to the question if it can be found in the data.
        SENTIMENT: Returns a sentiment score, from -1 to 1, representing the detected positive or negative sentiment of the given text.
        SUMMARIZE: Returns a summary of the given text.
        TRANSLATE: Translates given text from any supported language to any other.
    Required Privileges
    The CORTEX_USER database role in the SNOWFLAKE database includes the privileges that allow users to call Snowflake Cortex LLM functions. By default, the CORTEX_USER role is granted to the PUBLIC role. 
    The PUBLIC role is automatically granted to all users and roles, so this allows all users in your account to use the Snowflake Cortex LLM functions.
    If you don't want all users to have this privilege, you can revoke access to the PUBLIC role and grant access to specific roles.
    To revoke the CORTEX_USER database role from the PUBLIC role, run the following command using the ACCOUNTADMIN role:
    """
    labels_2 = ["programming language", "company", "command", "product"]

    text_3 = """
    Die Türkei und China haben anlässlich eines Besuchs des türkischen Außenministers Hakan Fidan in Peking über eine mögliche Mitgliedschaft der Türkei in der BRICS-Gruppe gesprochen. Dies ist eine weitere Entwicklung in Richtung multipolarer Weltordnung, analysiert Politik-Experte Berthold Kuhn.
    Was könnte die Motivation der Türkei sein, Mitglied von BRICS werden zu wollen?
    Die Motivation der Türkei, Mitglied von BRICS zu werden, würde das Streben der Türkei nach einem „neuen Platz“ in der sich verändernden Weltordnung und ihren wirtschaftspolitischen Pragmatismus unterstreichen. Durch eine Mitgliedschaft könnte die Türkei ihre Wirtschaft breiter aufstellen und sich weniger anfällig für externe Schocks machen. Zudem könnte sie ihren geopolitischen Einfluss erweitern.
    Seit seiner Gründung im Jahr 2006 durch Brasilien, Russland, Indien und China hat sich BRICS zu einem bedeutenden Bündnis von aufstrebenden Volkswirtschaften entwickelt. Angesichts der wirtschaftlichen Stagnation in den Industrienationen haben Investoren vermehrt ihr Augenmerk auf diese Staaten gerichtet, was zu einer verstärkten Emission von Finanzprodukten führte. Die Erweiterung um Südafrika im Jahr 2010 und jüngst um Ägypten, Äthiopien, Iran und die Vereinigten Arabischen Emirate zu Beginn des Jahres 2024 unterstreicht die wachsende Bedeutung des Bündnisses. Diese Erweiterung hat dazu geführt, dass in der internationalen Diplomatie von "BRICS plus" die Rede ist. Obwohl Argentinien zunächst Interesse bekundete, hat das Land nach der Wahl eines neuen libertären Präsidenten entschieden, dem Bündnis nicht beizutreten. Speziell China ist an der Ausweitung und Stärkung des Bündnisses interessiert, um sein wirtschaftliches Gewicht auch weltpolitisch stärker einbringen zu können.
    BRICS hat sich als bedeutende Gruppe auf der globalen Bühne etabliert, die alternative wirtschaftspolitische zu westlich dominierten Institutionen wie der Weltbank und dem Internationalen Währungsfonds (IWF) anbietet. Die Türkei könnte ihre Souveränität durch mehr Unabhängigkeit von westlichen politischen und wirtschaftlichen Staaten stärken wollen und durch die Zusammenarbeit mit BRICS-Staaten Alternativen zu westlich dominierten Finanzsystemen und internationalen Regeln finden. Darüber hinaus könnten Investitionen und Handel gefördert werden, da BRICS-Mitglieder oft an gemeinsamen Entwicklungsprojekten arbeiten und gegenseitig in ihre Volkswirtschaften investieren.
    Technologie- und Wissenstransfer könnten ebenfalls eine Rolle spielen. Eine Mitgliedschaft in BRICS könnte der Türkei Zugang zu technologischen Innovationen und Know-how aus den Mitgliedsländern verschaffen, insbesondere aus China und Indien, die führend in Bereichen wie Technologie und Wissenschaft sind. Schließlich verfügt BRICS über Institutionen wie die Neue Entwicklungsbank (NDB), die Infrastrukturprojekte und Entwicklungsinitiativen in Mitgliedsländern finanziert. Die Türkei könnte von solchen Finanzierungen profitieren, um ihre eigene wirtschaftliche Entwicklung zu fördern.
    Dr. Berthold Kuhn, Politikwissenschaftler, wurde an der Universität Leipzig promoviert und an der FU Berlin habilitiert. Kuhn arbeitet mit mehreren Universitäten in Europa und Asien als Experte für nachhaltige Entwicklung und internationale Beziehungen zusammen und berät die EU Kommission, internationale Organisationen und Denkfabriken. Er lebt aktuell in Xiamen (ggü. Taiwan) und in Berlin. Er ist Co-Autor des Buchs „ Global Perspectives on Megatrends“ (Ibidem, Columbia University Press).
    Welche Auswirkungen könnte eine Mitgliedschaft der Türkei in BRICS auf die europäische Union und die globale Wirtschaft haben?
    Die mögliche Mitgliedschaft der Türkei in BRICS würde sowohl die Europäische Union (EU) als auch die globale Wirtschaft beeinflussen. Geopolitisch gesehen, könnte sich die Türkei stärker auf die BRICS-Staaten ausrichten, was die Dynamik der EU-Türkei-Beziehungen verändern und möglicherweise zu erhöhten Spannungen führen könnte. Auf wirtschaftlicher Ebene könnten engere Bindungen der Türkei an BRICS-Staaten die Handelsbeziehungen zur EU beeinflussen. Es besteht die Möglichkeit, dass türkische Unternehmen verstärkt in BRICS-Länder exportieren und importieren, was zu einer Veränderung der Handelsströme führen könnte. Zudem könnte die Türkei durch eine BRICS-Mitgliedschaft an Einfluss in globalen Institutionen gewinnen, was auch Auswirkungen auf die EU und die Außenpolitik der EU Mitgliedstaaten hätte.
    In Bezug auf die globale Wirtschaft könnten Investitionen und wirtschaftliche Zusammenarbeit mit den BRICS-Staaten das Wirtschaftswachstum der Türkei ankurbeln und zu einer stärkeren Integration in die globale Wirtschaft führen. Eine engere Anbindung an BRICS könnte Auswirkungen auf die globalen Finanzmärkte haben, insbesondere bei einer Zunahme von Finanztransaktionen und Handel in den Währungen der BRICS-Staaten. Darüber hinaus könnten neue Handelsallianzen geschmiedet werden, was Druck auf bestehende Handelsabkommen und wirtschaftliche Partnerschaften ausüben könnte.
    Eine intensivere Zusammenarbeit zwischen der Türkei und den rohstoffreichen BRICS-Staaten könnte zudem die globalen Rohstoffmärkte beeinflussen. Zusammenfassend lässt sich sagen, dass eine Mitgliedschaft der Türkei in BRICS signifikante geopolitische und wirtschaftliche Veränderungen mit sich bringen würde, welche sowohl die EU als auch die globale Wirtschaft beeinflussen könnten.
    Welche Vorteile könnte eine BRICS-Mitgliedschaft für die Türkei gegenüber einer EU-Mitgliedschaft bieten?
    Eine Mitgliedschaft in der BRICS-Gruppe könnte der Türkei verschiedene Vorteile bieten, die sich von denen einer EU-Mitgliedschaft unterscheiden. Zudem erscheint eine Mitgliedschaft der Türkei in der EU, speziell unter der Amtsführung von Präsident Erdogan, nicht realistisch. Auch ist in vielen EU-Ländern die öffentliche Unterstützung für einen türkischen Beitritt gering, was EU-seitig weitere Verhandlungen nicht gerade begünstigt. Es gibt insgesamt grundlegende politische, auch wirtschaftspolitische, Differenzen.
    Während die EU eine supranationale Organisation mit umfangreichen politischen, wirtschaftlichen und rechtlichen Integrationsmechanismen ist, handelt es sich bei BRICS um ein lockeres Bündnis aufstrebender Volkswirtschaften, das auf wirtschaftlicher Zusammenarbeit und politischem Dialog basiert.
    Die BRICS-Länder zeichnen sich durch wirtschaftlichen Pragmatismus und Flexibilität aus und legen ihren Mitgliedern weniger Vorgaben und Regulierungen auf als die EU. Dies könnte der Türkei erhebliche Investitionsmöglichkeiten und Finanzierungen durch Institutionen wie die Neue Entwicklungsbank (NDB) bieten und so helfen, große Infrastrukturprojekte und wirtschaftliche Entwicklungsinitiativen voranzutreiben. Darüber hinaus könnte eine BRICS-Mitgliedschaft der Türkei helfen, ihre geopolitische Unabhängigkeit von westlichen Einflussbereichen zu stärken und ihre Rolle als eigenständiger Akteur auf der globalen Bühne zu festigen.
    Sie könnte auch Zugang zu den aufstrebenden Märkten der BRICS-Staaten erhalten, was den Handel und die wirtschaftliche Zusammenarbeit mit diesen dynamischen Volkswirtschaften fördern könnte. Eine engere Zusammenarbeit mit technologisch fortschrittlichen BRICS-Ländern wie China und Indien könnte der Türkei Zugang zu neuen Technologien und Innovationen verschaffen. Zudem könnte sie ihre nationale Souveränität besser wahren, da BRICS-Staaten tendenziell weniger in die innenpolitischen Angelegenheiten ihrer Mitglieder eingreifen als die EU.
    Schließlich könnte die Türkei in BRICS ein Forum finden, um gemeinsame Interessen und Herausforderungen mit anderen aufstrebenden Volkswirtschaften zu diskutieren und anzugehen, was ihre internationale Verhandlungsposition stärken könnte. Im Gegensatz dazu stellt eine EU-Mitgliedschaft hohe Anforderungen an wirtschaftliche und politische Reformen, die zu einer stärkeren Integration führen.
    Welche nächsten Schritte könnten wir erwarten, wenn die Türkei ernsthaft eine BRICS-Mitgliedschaft in Betracht zieht?
    Sollte die Türkei tatsächlich eine Mitgliedschaft in der BRICS-Gruppe anstreben, könnten die nächsten Schritte in etwa folgendermaßen aussehen: Zunächst hat die Türkei den Besuch ihres Außenministers in Peking strategisch genutzt, um die Idee einer BRICS-Mitgliedschaft ins Spiel zu bringen. Sie strebt engere wirtschaftspolitische Beziehungen zu China und den anderen Mitgliedstaaten an, wobei sie sich der außenpolitischen Risiken in Bezug auf Russland bewusst ist. Daher ist auf dem bevorstehenden BRICS-Gipfel in Russland im Oktober noch keine Entscheidung zu erwarten. Der Kreml hat jedoch bereits Interesse signalisiert, das Thema auf die Agenda des BRICS-Gipfels 2024 zu setzen.
    Auch wird die Türkei die Folgen der Wahlergebnisse in Indien und in Südafrika beobachten wollen. Speziell in Südafrika hat die Regierungspartei, der African National Congress, erhebliche Einbissen hinnehmen müssen. In Indien wird Premier Modi weiter regieren können, aber sein Koalitionsbündnis hat empfindliche Einbußen hinnehmen müssen. Die Türkei würde intensive diplomatische Gespräche mit den bestehenden BRICS-Mitgliedern führen, um Unterstützung für ihre Mitgliedschaft zu gewinnen.
    Parallel dazu könnte die türkische Regierung öffentliches Interesse und Engagement für die BRICS-Gruppe signalisieren, um ihren politischen Willen und ihre Entschlossenheit zu demonstrieren. Darüber hinaus könnte die Türkei einen Beobachterstatus bei BRICS-Treffen anstreben und an wichtigen Diskussionen teilnehmen, um Beziehungen zu den Mitgliedsländern zu stärken. Gleichzeitig könnte sie wirtschaftliche und handelspolitische Maßnahmen ergreifen, um ihre Wirtschaft stärker mit den BRICS-Staaten zu verflechten und dadurch ihre Attraktivität als potenzielles Mitglied zu erhöhen.
    Die Türkei könnte auch bilaterale Abkommen mit BRICS-Staaten abschließen, um wirtschaftliche und politische Zusammenarbeit zu vertiefen. Sie würde ihre Außenpolitik anpassen und eine stärkere Ausrichtung auf BRICS-Themen in internationalen Foren zeigen. Schließlich würde die Türkei formell einen Antrag auf Mitgliedschaft einreichen und den formalen Prozess zur Aufnahme in die BRICS-Gruppe durchlaufen. Diese Schritte würden zeigen, dass die Türkei es ernst meint mit ihrer Absicht, BRICS-Mitglied zu werden, und bereit ist, sich aktiv in die Gruppe zu integrieren.
    Was passiert, wenn die Abgabe der Steuererklärung vergessen wird? Steueranwalt Stefan Heine erklärt, wann welche Strafen drohen und was Sie jetzt noch unternehmen können, wenn Sie die Abgabe absehbar nicht bis zum Stichtag schaffen.
    Das Teilen privater Fotos in sozialen Medien gefällt auch der politischen Elite. Doch warum entscheiden sich Personen wie Ricarda Lang dafür, obwohl sie dadurch angreifbar werden? Life Coach Chris Oeuvray analysiert dieses Verhalten und erklärt, welche Rolle Authentizität und Transparenz spielen.
    Wie wäre eine BRICS-Mitgliedschaft der Türkei geopolitisch zu bewerten?
    Eine mögliche BRICS-Mitgliedschaft der Türkei wäre ein bedeutender geopolitischer Schritt, der die Entwicklungen hin zu einer multipolaren Weltordnung und einer pragmatischen Ausrichtung wirtschaftspolitischer Beziehungen von Schwellenländern verdeutlichen würde. Sie würde die geopolitische Landschaft verändern und die Rolle der Türkei auf der globalen Bühne stärken.
    Allerdings gäbe es auch strategische und diplomatische Herausforderungen, insbesondere für die Türkei selbst, die EU und die NATO. Die Türkei stünde in Zeiten geopolitischer Krisen und geoökonomischer Konflikte ständig vor Herausforderungen, eine Balance zwischen den Vorteilen neuer Partnerschaften und den bestehenden Verpflichtungen gegenüber westlichen Institutionen zu finden. Eine solche Mitgliedschaft könnte Spannungen mit der EU und der NATO hervorrufen, da sich die Türkei enger an nicht-westliche Mächte wie China und Indien bindet und in einen Staatenbund eintritt, in dem auch Russland eine wichtige Rolle spielt.
    Die Sorge um verstärkte autoritäre politische Tendenzen in der Türkei und um BRICS als Gegengewicht zu westlich dominierten Institutionen könnte zunehmen. Zudem könnte die Türkei den Einfluss von BRICS in internationalen Institutionen wie der UNO und dem IWF stärken. Wirtschaftlich gesehen könnte die Türkei von neuen Märkten und Investitionen sehr wahrscheinlich profitieren, insbesondere durch engere Beziehungen zu China und Indien. Dennoch müsste sie eine Balance zwischen ihren traditionellen westlichen Verbündeten und den neuen BRICS-Partnerschaften finden, was komplexe diplomatische Herausforderungen, speziell gegenüber den USA der EU und der NATO, mit sich bringen würde.
    Dieser Text stammt von einem Expert aus dem FOCUS online EXPERTS Circle. Unsere Experts verfügen über hohes Fachwissen in ihrem Themenbereich und sind nicht Teil der Redaktion. Mehr erfahren.
    """
    labels_3 = ["country", "organization", "person"]

    for text, labels in ((text_1, labels_1), (text_2, labels_2), (text_3, labels_3)):
        chunks = split_into_combined_sentence_chunks(text)
        entities = EntityFinder(labels).find(*chunks)
        for entity in entities:
            print(entity.name, "=>", entity.label)
