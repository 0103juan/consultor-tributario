"""Consultor tributario: reescritura de la pregunta -> recuperación híbrida + rerank -> respuesta citada -> verificación.

    python pipeline.py "¿Cuánto es la sanción por presentar la declaración tarde?"
"""

import json
import re
import sys
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel

from retrieval import DATA, Chunk, Index

# A safety decline is re-run server-side on Anthropic's recommended fallback model.
LLM = {"model": "claude-sonnet-5-5", "max_tokens": 16000,
       "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
ABSTAIN = "No lo sé con base en el Estatuto Tributario."

REWRITE_SYSTEM = """Preparas consultas de búsqueda sobre el Estatuto Tributario de Colombia.

A partir del historial y de la última pregunta, produce:
- standalone: la pregunta reescrita para que se entienda sin el historial. Resuelve pronombres y preguntas \
elípticas ("¿y si la corrijo?") con el historial. Si ya se entiende sola, déjala igual.
- variants: dos reformulaciones con el vocabulario que usaría la norma. Cambia los nombres coloquiales por \
los legales (por ejemplo "4x1000" -> "gravamen a los movimientos financieros", "multa" -> "sanción", \
"información exógena" -> "obligación de suministrar información", "IVA" -> "impuesto sobre las ventas"). \
Conserva tal cual los números de artículo."""

GENERATE_SYSTEM = f"""Respondes preguntas sobre el Estatuto Tributario de Colombia usando únicamente los pasajes \
numerados del contexto. Cada pasaje es un artículo o una parte de un artículo.

- Cita el pasaje que respalda cada frase con su número entre corchetes, como [2], y nombra el artículo \
("según el artículo 641 [1]").
- Puedes dar un paso de inferencia evidente si lo dices ("una SAS es una sociedad nacional, así que..."). \
No encadenes varios.
- Deja las cifras en UVT. No las conviertas a pesos: el valor de la UVT cambia cada año y no está en el contexto.
- Un pasaje marcado como DEROGADO ya no rige. Úsalo solo para decir que la norma fue derogada y por cuál.
- Si la respuesta depende de una tabla o fórmula marcada como publicada como imagen, di que no está disponible \
como texto y remite a la fuente oficial. No reconstruyas la tabla de memoria.
- Responde en dos o tres frases, sin recomendaciones sobre el caso particular de quien pregunta.
- Si los pasajes no contienen la respuesta, responde exactamente esta frase y nada más: {ABSTAIN}"""

JUDGE_SYSTEM = """Auditas una respuesta contra los pasajes de la norma de los que se escribió.

Divide la respuesta en afirmaciones fácticas atómicas, sin contar las marcas de cita. Marca una afirmación \
como respaldada solo si los pasajes la dicen o se sigue directamente de ellos. El conocimiento externo a los \
pasajes no cuenta, aunque sea cierto: una cifra, un plazo o un número de artículo que no aparezca en los \
pasajes es una afirmación sin respaldo."""


class Rewrite(BaseModel):
    standalone: str
    variants: list[str]


class Claim(BaseModel):
    text: str
    supported: bool


class Verdict(BaseModel):
    claims: list[Claim]


@dataclass
class Result:
    answer: str
    queries: list[str]
    retrieved: list[Chunk]
    sources: list[Chunk]  # the retrieved passages the answer actually cites
    claims: list[Claim]

    @property
    def unsupported(self) -> list[str]:
        return [c.text for c in self.claims if not c.supported]

    @property
    def groundedness(self) -> float:
        return 1 - len(self.unsupported) / len(self.claims) if self.claims else 1.0


def _context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{n}] {chunk.text}" for n, chunk in enumerate(chunks, 1))


def rewrite(client, question: str, history: str = "") -> list[str]:
    response = client.beta.messages.parse(
        **LLM, system=REWRITE_SYSTEM, output_config={"effort": "low"}, output_format=Rewrite,
        messages=[{"role": "user",
                   "content": f"<historial>\n{history}\n</historial>\n\n<pregunta>\n{question}\n</pregunta>"}])
    parsed = response.parsed_output
    if parsed is None:  # declined or truncated: retrieval still works on the raw question
        return [question]
    return [parsed.standalone, *parsed.variants[:2]]


def generate(client, question: str, chunks: list[Chunk], correction: str = "") -> str:
    response = client.beta.messages.create(
        **LLM, system=GENERATE_SYSTEM, output_config={"effort": "low"},
        messages=[{"role": "user", "content":
                   f"<contexto>\n{_context(chunks)}\n</contexto>\n\n<pregunta>\n{question}\n</pregunta>{correction}"}])
    if response.stop_reason == "refusal":
        return ABSTAIN
    return "".join(block.text for block in response.content if block.type == "text").strip()


def judge(client, answer: str, chunks: list[Chunk]) -> list[Claim]:
    response = client.beta.messages.parse(
        **LLM, system=JUDGE_SYSTEM, output_config={"effort": "medium"}, output_format=Verdict,
        messages=[{"role": "user", "content":
                   f"<pasajes>\n{_context(chunks)}\n</pasajes>\n\n<respuesta>\n{answer}\n</respuesta>"}])
    if response.parsed_output is None:  # fail closed: an answer we could not verify is not grounded
        return [Claim(text="(verificación no disponible)", supported=False)]
    return response.parsed_output.claims


def answer(client, index: Index, question: str, history: str = "") -> Result:
    queries = rewrite(client, question, history)
    chunks = index.search(queries)

    def attempt(correction: str = "") -> tuple[str, list[Claim]]:
        text = generate(client, queries[0], chunks, correction)
        return text, ([] if text == ABSTAIN else judge(client, text, chunks))

    text, claims = attempt()
    unsupported = [c.text for c in claims if not c.supported]
    if unsupported:  # one corrective pass; if it still fails, the caller sees result.unsupported
        text, claims = attempt(
            "\n\nUn borrador anterior hizo estas afirmaciones, que los pasajes no respaldan. "
            "No las incluyas:\n" + "\n".join(f"- {claim}" for claim in unsupported))

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    sources = [chunks[n - 1] for n in cited if 1 <= n <= len(chunks)]
    return Result(text, queries, chunks, sources, claims)


if __name__ == "__main__":
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    result = answer(anthropic.Anthropic(), Index.load(), " ".join(sys.argv[1:]))
    print(result.answer, "\n")
    for source in result.sources:
        print(f"  fuente: {source.title[:80]}\n          {source.url}")
    print(f"  respaldo de las afirmaciones: {result.groundedness:.0%}")
    for claim in result.unsupported:
        print(f"  SIN RESPALDO: {claim}")
    print(f"\nTexto según la compilación del Senado actualizada al {meta['compilation_updated']}. "
          "Esto no es asesoría tributaria.")
