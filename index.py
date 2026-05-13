import os
import re
import math
import random
from dotenv import load_dotenv
from collections import Counter
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

APP_TITLE = "IA For HEALTH - Santa Casa BH"
ENV_FILE = "dados.env"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(BASE_DIR, "resultados.txt")
KNOWLEDGE_FILE = "rag_santa_casa_bh_2024.txt"
KNOWLEDGE_DIR = "knowledge_base"
RAG_CHUNK_SIZE = 900
RAG_CHUNK_OVERLAP = 180
RAG_TOP_K = 4
APP_VERSION = "v7.2 - 04/05/2026 - RAG + OpenAI"


# =============================
# ENVler local KEY
# =============================
""" 
def load_environment():
    load_dotenv(ENV_FILE)
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY não encontrada.")
        st.stop()
"""


# =============================
# ENV LER VIA SECRET STREMIT
# =============================
def load_environment():
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    except Exception:
        st.error("OPENAI_API_KEY não encontrada nas Secrets do Streamlit.")
        st.stop()

# =============================
# MODELO
# =============================
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)


# =============================
# RAG SIMPLES LOCAL
# =============================
def read_text_file(path: str) -> str:
    """Lê arquivo TXT aceitando UTF-8 e Latin-1."""
    try:
        with open(path, "r", encoding="utf-8") as arquivo:
            return arquivo.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as arquivo:
            return arquivo.read().strip()


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-zà-ú0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    stopwords = {
        "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do", "das", "dos",
        "e", "em", "no", "na", "nos", "nas", "para", "por", "com", "sem", "sobre", "que",
        "qual", "quais", "quem", "quando", "onde", "como", "porque", "porquê", "é", "ser",
        "são", "foi", "sua", "seu", "suas", "seus", "ao", "à", "aos", "às", "me", "minha",
        "meu", "tem", "ter", "pode", "posso", "vou", "vai", "santa", "casa", "bh"
    }
    words = normalize_text(text).split()
    return [w for w in words if len(w) >= 3 and w not in stopwords]


def unique_list(items):
    """Remove duplicidades mantendo a ordem original."""
    unique = []
    seen = set()
    for item in items:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def remove_fonte_lines(answer: str) -> str:
    """Remove linhas de fonte geradas pelo modelo para evitar duplicidade."""
    if not answer:
        return answer

    lines = answer.splitlines()
    clean_lines = []
    for line in lines:
        if line.strip().lower().startswith("fonte:"):
            continue
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()

def chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP):
    """Divide o texto em trechos com sobreposição para melhorar a recuperação."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def get_knowledge_paths():
    """Procura arquivos .txt no knowledge_base e também o rag_santa_casa_bh_2024.txt na raiz."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        os.path.join(base_dir, KNOWLEDGE_DIR),
        os.path.join(os.getcwd(), KNOWLEDGE_DIR),
        KNOWLEDGE_DIR,
    ]

    paths = []

    root_candidates = [
        os.path.join(base_dir, KNOWLEDGE_FILE),
        os.path.join(os.getcwd(), KNOWLEDGE_FILE),
        KNOWLEDGE_FILE,
    ]

    for path in root_candidates:
        if os.path.isfile(path) and path not in paths:
            paths.append(path)

    for folder in candidate_dirs:
        if os.path.isdir(folder):
            for file_name in os.listdir(folder):
                if file_name.lower().endswith(".txt"):
                    full_path = os.path.join(folder, file_name)
                    if full_path not in paths:
                        paths.append(full_path)

    return paths


@st.cache_data(show_spinner=False)
def build_rag_index():
    """Cria um índice local simples com chunks dos arquivos TXT."""
    docs = []
    paths = get_knowledge_paths()

    for path in paths:
        try:
            text = read_text_file(path)
        except Exception:
            continue

        if not text:
            continue

        for idx, chunk in enumerate(chunk_text(text)):
            docs.append({
                "source": os.path.basename(path),
                "chunk_id": idx + 1,
                "text": chunk,
                "tokens": tokenize(chunk),
            })

    return docs


def retrieve_context(question: str, top_k: int = RAG_TOP_K):
    """Busca os trechos mais relevantes usando pontuação lexical com IDF."""
    docs = build_rag_index()
    if not docs:
        return "", []

    query_tokens = tokenize(question)
    if not query_tokens:
        return "", []

    total_docs = len(docs)
    document_frequency = Counter()
    for doc in docs:
        for token in set(doc["tokens"]):
            document_frequency[token] += 1

    query_counter = Counter(query_tokens)
    scored_docs = []

    for doc in docs:
        chunk_counter = Counter(doc["tokens"])
        score = 0.0

        for token, query_count in query_counter.items():
            if token in chunk_counter:
                idf = math.log((1 + total_docs) / (1 + document_frequency[token])) + 1
                score += query_count * chunk_counter[token] * idf

        normalized_question = normalize_text(question)
        normalized_chunk = normalize_text(doc["text"])
        if normalized_question and normalized_question in normalized_chunk:
            score += 5

        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    selected = [doc for _, doc in scored_docs[:top_k]]

    context_parts = []
    sources = []
    used_labels = set()

    for doc in selected:
        label = f"RAG_SantaCasaBH - trecho {doc['chunk_id']}"

        # Evita repetir a mesma fonte quando o mesmo trecho aparece mais de uma vez.
        if label in used_labels:
            continue

        used_labels.add(label)
        sources.append(label)
        context_parts.append(f"[Fonte interna: {label}]\n{doc['text']}")

    return "\n\n---\n\n".join(context_parts), unique_list(sources)


def load_knowledge_base():
    """Compatibilidade: retorna todo o conteúdo indexado, se necessário."""
    docs = build_rag_index()
    return "\n\n".join([doc["text"] for doc in docs])

def render_banner_topo_chat():
    if not st.session_state.quiz_completed:
        return

    
# =============================
# MODELOS
# =============================
class AnswerEvaluation(BaseModel):
    correct: bool
    score: int
    feedback: str

# =============================
# UNIDADES_NEGOCIO
# =============================

UNIDADES_NEGOCIO = [    
    "Santa Casa BH",
    "São Lucas", 
    "Centro de Autismo",
    "Funerária e Assistência Familiar",
    "Faculdade de Saúde",
    "Ambulatórios Especializados",
    "Instituto Geriátrico",
    "Instituto Materno Infantil",
    "Instituto de Oncologia",
    "Pesquisa Clínica",
    "Órix Lab",
    "Corporativo"
]


# =============================
# PERGUNTAS
# =============================
QUESTION_BANK = [
    {
        "question": "O que é inteligência artificial?",
        "reference": "Simulação da inteligência humana, máquinas, aprendizado, decisão IA é a capacidade de sistemas computacionais simularem a inteligência humana, realizando tarefas como aprendizado, reconhecimento de padrões e tomada de decisão."
    },
    {
        "question": "O que é aprendizado de máquina (Machine Learning)?",
        "reference": "Aprendizado com dados, padrões, sem programação explícita. É uma área da IA em que sistemas aprendem a partir de dados, identificando padrões e melhorando seu desempenho sem serem explicitamente programados para cada tarefa."
    },
    {
        "question": "Qual a diferença entre aprendizado supervisionado e não supervisionado?",
        "reference": "supervisionado: dados rotulados não supervisionado: padrões sem rótulo. No aprendizado supervisionado, o modelo é treinado com dados rotulados (com respostas conhecidas). No não supervisionado, o modelo identifica padrões em dados sem rótulos."
    },
    {
        "question": "O que são dados de treinamento em IA?",
        "reference": "base de aprendizado, exemplos históricos. São os dados utilizados para treinar um modelo de IA, permitindo que ele aprenda padrões e relações para executar determinada tarefa."
    },
    {
        "question": "O que significa dizer que um modelo “aprende com dados”?",
        "reference": "padrões, ajuste de parâmetros. Significa que o modelo analisa dados, identifica padrões e ajusta seus parâmetros internos para melhorar suas previsões ou respostas."
    },
    {
        "question": "O que é um algoritmo em IA?",
        "reference": "regras, instruções, processamento. É um conjunto de instruções ou regras que orienta o sistema a processar dados e executar uma tarefa."
    },
    {
        "question": "O que é um LLM?",
        "reference": "modelo de linguagem, grande volume de dados, texto. É um modelo de IA treinado com grandes volumes de texto, capaz de compreender e gerar linguagem natural."
    },
    {
        "question": "O que é IA generativa?",
        "reference": "criação de conteúdo, texto/imagem. É um tipo de IA capaz de gerar novos conteúdos, como textos, imagens ou códigos, com base em padrões aprendidos."
    },
    {
        "question": "O que é um modelo de linguagem?",
        "reference": "previsão de palavras, sequência. É um modelo que aprende a prever e gerar sequências de palavras com base no contexto."
    },
    {
        "question": "O que é RAG?",
        "reference": "busca + geração, base externa. É uma técnica que combina recuperação de informações em bases externas com geração de respostas por IA, tornando-as mais precisas e atualizadas."
    },
    {
        "question": "O que é um modelo pré-treinado?",
        "reference": "treinado previamente, reutilização. É um modelo já treinado em grande volume de dados, que pode ser reutilizado ou adaptado para novas tarefas."
    },
    {
        "question": "O que é inferência em IA?",
        "reference": "uso do modelo, resposta. É o momento em que o modelo já treinado é utilizado para gerar previsões ou respostas."
    },
    {
        "question": "O que é um prompt?",
        "reference": "comando, entrada. É a instrução ou pergunta fornecida ao modelo de IA para gerar uma resposta."
    },
    {
        "question": "O que caracteriza um bom prompt para IA?",
        "reference": "clareza, contexto, objetivo. Um bom prompt é claro, específico e fornece contexto suficiente para orientar a IA a gerar uma resposta relevante."
    },
    {
        "question": "O que é engenharia de prompt?",
        "reference": "otimização, melhoria de resposta. É a prática de criar e ajustar prompts para obter melhores resultados dos modelos de IA."
    },
    {
        "question": "Por que a mesma pergunta feita para a IA pode gerar respostas diferentes?",
        "reference": "variação, contexto, aleatoriedade. Porque os modelos podem gerar respostas diferentes com base em variações internas, contexto e aleatoriedade no processo de geração."
    },
    {
        "question": "O que é viés (bias)?",
        "reference": "distorção, dados enviesados. É quando o modelo apresenta resultados distorcidos ou injustos devido a dados de treinamento enviesados."
    },
    {
        "question": "O que são alucinações?",
        "reference": "informação incorreta, invenção. São respostas geradas pela IA que parecem corretas, mas são falsas ou não têm base real."
    },
    {
        "question": "Por que a IA pode gerar informações incorretas?",
        "reference": "dados limitados, contexto. Porque o modelo depende dos dados de treinamento e pode não ter informações completas ou interpretar o contexto de forma incorreta."
    },
    {
        "question": "Quais são os riscos de usar IA na saúde?",
        "reference": "erro clínico, privacidade, LGPD. Os riscos incluem decisões incorretas, vazamento de dados sensíveis, dependência excessiva e impactos na segurança do paciente."
    },
    {
        "question": "Por que não usar dados de pacientes em IA pública?",
        "reference": "vazamento, segurança. Porque pode haver risco de vazamento e uso indevido de dados, violando leis de proteção de dados."
    },
    {
        "question": "O que é uso responsável de IA?",
        "reference": "ética, supervisão. É o uso da IA com responsabilidade, garantindo segurança, ética e supervisão humana."
    },
    {
        "question": "De quem é a responsabilidade por decisões com IA?",
        "reference": "humano responsável. A responsabilidade continua sendo do profissional humano, pois a IA é apenas uma ferramenta de apoio."
    },
    
]


# =============================
# ESTADO
# =============================
def init_state():
    defaults = {
        "started": False,
        "phase": "idle",  # idle, name, matricula, unidade, quiz, extra, chat
        "name": "",
        "matricula": "",
        "unidade_negocio": "",
        "aceitou_responder_quiz": "",
        "questions": [],
        "index": 0,
        "results": [],
        "score": 0,
        "level": "",
        "chat": [],
        "extra_question_needed": False,
        "extra_question_answered": False,
        "extra_question_answer": "",
        "extra_question_topic": "",
        "focus_input": False,
        "scroll_to_result": False,
        "scroll_to_bottom": False,
        "quiz_completed": False,
        "test_start_time": None,
        "test_end_time": None,
        "result_saved": False,
        "extra_question_processing": False,
        "pending_scroll_bottom": False,
        "scroll_nonce": 0,
        "force_result_scroll_after_extra": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================
# UTIL
# =============================
def add_msg(role, text, section="quiz"):
    st.session_state.chat.append({
        "role": role,
        "content": text,
        "section": section
    })


def classify(score):
    if score <= 2:
        return "Básico"
    elif score == 3:
        return "Intermediário"
    return "Avançado"


def get_level_class(level):
    level_normalized = (level or "").strip().lower()
    if level_normalized == "básico" or level_normalized == "basico":
        return "level-basico"
    if level_normalized == "intermediário" or level_normalized == "intermediario":
        return "level-intermediario"
    if level_normalized == "avançado" or level_normalized == "avancado":
        return "level-avancado"
    return "level-intermediario"


def format_datetime(value):
    if not value:
        return "-"
    return value.strftime("%d/%m/%Y %H:%M:%S")


def save_result_to_file():
    if not st.session_state.quiz_completed or st.session_state.result_saved:
        return

    # Só bloqueia se realmente precisa da pergunta extra
    if (
        st.session_state.level in ["Intermediário", "Avançado"]
        and st.session_state.extra_question_needed
        and not st.session_state.extra_question_answered
    ):
        return

    if st.session_state.test_end_time is None:
        st.session_state.test_end_time = datetime.now()

    linhas = [
        f"Data Hora início: {format_datetime(st.session_state.test_start_time)}",
        f"Nome: {st.session_state.name}",
        f"Matrícula: {st.session_state.matricula}",
        f"Unidade de Negócio: {st.session_state.unidade_negocio}",
        f"Nível: {st.session_state.level}",
        f"Pontuação: {st.session_state.score}/5",
    ]

    if st.session_state.level in ["Intermediário", "Avançado"]:
        linhas.append(f"Deseja compartilhar conhecimentos: {st.session_state.extra_question_answer}")

    if st.session_state.extra_question_answer == "Sim":
        linhas.append(f"Tema: {st.session_state.extra_question_topic}")

    linhas.append(f"Data Hora Final: {format_datetime(st.session_state.test_end_time)}")
    linhas.append("-" * 60)

    with open(RESULT_FILE, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas) + "\n")

    st.session_state.result_saved = True


def render_tema_compartilhamento_form():
    if st.session_state.phase != "tema_compartilhamento":
        return

    st.markdown(
        '<div class="result-section-title">Quais temas você deseja compartilhar?</div>',
        unsafe_allow_html=True
    )

    temas = st.multiselect(
        "Selecione um ou mais temas:",
        TEMAS_COMPARTILHAMENTO,
        key="input_tema_compartilhamento"
    )

    if st.button("💾 Salvar temas", key="btn_salvar_tema_compartilhamento"):
        if not temas:
            st.warning("Selecione pelo menos um tema antes de salvar.")
            return

        process_tema_compartilhamento(temas)

def reset_test_data():
    st.session_state.phase = "name"
    st.session_state.chat = []
    st.session_state.results = []
    st.session_state.index = 0
    st.session_state.score = 0
    st.session_state.level = ""
    st.session_state.name = ""
    st.session_state.matricula = ""
    st.session_state.unidade_negocio = ""
    st.session_state.questions = []
    st.session_state.extra_question_needed = False
    st.session_state.extra_question_answered = False
    st.session_state.extra_question_answer = ""
    st.session_state.extra_question_topic = ""
    st.session_state.focus_input = True
    st.session_state.scroll_to_result = False
    st.session_state.scroll_to_bottom = True
    st.session_state.quiz_completed = False
    st.session_state.test_start_time = datetime.now()
    st.session_state.test_end_time = None
    st.session_state.result_saved = False
    st.session_state.extra_question_processing = False
    st.session_state.pending_scroll_bottom = False
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1
    st.session_state.force_result_scroll_after_extra = False


def start_conversation():
    st.session_state.started = True
    reset_test_data()
    add_msg("assistant", "Olá! Qual é o seu nome?", section="quiz")


def voltar_resultado():
    st.session_state.focus_input = False
    st.session_state.scroll_to_result = True
    st.session_state.scroll_to_bottom = False
    st.session_state.pending_scroll_bottom = False
    st.session_state.force_result_scroll_after_extra = True
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1
    st.rerun()


def responder_pergunta_extra(resposta: str):
    st.session_state.extra_question_answer = resposta
    st.session_state.extra_question_processing = False
    st.session_state.focus_input = False
    st.session_state.test_end_time = datetime.now()

    add_msg("user", resposta, section="quiz")

    if resposta == "Sim":
        st.session_state.extra_question_answered = False
        st.session_state.phase = "tema_compartilhamento"
        add_msg(
            "assistant",
            "Qual tema você deseja abordar no ensinamento?",
            section="quiz"
        )
        st.session_state.scroll_to_bottom = True
        st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1
        st.session_state.focus_input = True
        st.rerun()
        return

    st.session_state.extra_question_answered = True
    st.session_state.phase = "chat"

    add_msg(
        "assistant",
        "Resposta registrada no card do resultado final. A partir de agora você pode usar o chat para tirar dúvidas sobre inteligência artificial e sobre a Santa Casa BH.",
        section="quiz"
    )

    st.session_state.focus_input = True
    st.session_state.scroll_to_bottom = True
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1

    save_result_to_file()
    st.rerun()

def process_tema_compartilhamento(temas):
    temas_formatados = "; ".join(temas)

    add_msg("user", temas_formatados, section="quiz")

    st.session_state.extra_question_topic = temas_formatados
    st.session_state.extra_question_answered = True
    st.session_state.phase = "chat"

    st.session_state.focus_input = True
    st.session_state.scroll_to_bottom = True
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1
    
    st.session_state.test_end_time = datetime.now()

    add_msg(
        "assistant",
        "Temas registrados com sucesso. A partir de agora você pode usar o chat para tirar dúvidas.",
        section="quiz"
    )

    save_result_to_file()
    st.rerun()


def render_footer_fixo():
    components.html("""
        <script>
        const doc = window.parent.document;

        function criarRodapeFixo() {
            let footer = doc.getElementById("footer-santa-ideia-fixo");

            if (!footer) {
                footer = doc.createElement("div");
                footer.id = "footer-santa-ideia-fixo";
                footer.innerHTML = "® Santa ideia 2026 - Órix Lab";
                doc.body.appendChild(footer);
            }

            footer.style.position = "fixed";
            footer.style.left = "0";
            footer.style.right = "0";
            footer.style.bottom = "4px";
            footer.style.width = "100%";
            footer.style.textAlign = "center";
            footer.style.color = "#64748b";
            footer.style.fontSize = "0.85rem";
            footer.style.fontWeight = "700";
            footer.style.zIndex = "2147483647";
            footer.style.pointerEvents = "none";
            footer.style.background = "transparent";
        }

        criarRodapeFixo();
        setInterval(criarRodapeFixo, 500);
        </script>
    """, height=0)


def reiniciar_teste():
    reset_test_data()
    add_msg("assistant", "Olá! Qual é o seu nome?", section="quiz")
    st.rerun()


# =============================
# SCRIPTS
# =============================
def run_focus_script():
    if not st.session_state.get("focus_input"):
        return

    phase = st.session_state.get("phase", "")

    if phase == "tema_compartilhamento":
        selectors_js = """
            [
                'input[aria-label="Informe o tema que deseja compartilhar:"]',
                'input[id*="input_tema_compartilhamento"]',
                'input[type="text"]'
            ]
        """
    else:
        selectors_js = """
            [
                'textarea[data-testid="stChatInputTextArea"]',
                'div[data-testid="stChatInput"] textarea',
                'div[data-testid="stChatInput"] input'
            ]
        """

    components.html(f"""
        <script>
        const focusInput = () => {{
            const doc = window.parent.document;
            const selectors = {selectors_js};

            for (const selector of selectors) {{
                const el = doc.querySelector(selector);
                if (el) {{
                    el.focus();
                    el.click();
                    if (el.setSelectionRange) {{
                        const len = el.value ? el.value.length : 0;
                        el.setSelectionRange(len, len);
                    }}
                    return true;
                }}
            }}
            return false;
        }};

        let tries = 0;
        const timer = setInterval(() => {{
            tries++;
            if (focusInput() || tries > 40) {{
                clearInterval(timer);
            }}
        }}, 150);
        </script>
    """, height=0)

    st.session_state.focus_input = False


def run_scroll_to_result_script():
    if not st.session_state.get("scroll_to_result") and not st.session_state.get("force_result_scroll_after_extra"):
        return

    nonce = st.session_state.get("scroll_nonce", 0)

    components.html(f"""
        <script>
        // execução forçada do scroll resultado: {nonce}

        function findScrollableParents(doc) {{
            const candidates = [
                doc.scrollingElement,
                doc.documentElement,
                doc.body,
                doc.querySelector('.stApp'),
                doc.querySelector('[data-testid="stAppViewContainer"]'),
                doc.querySelector('[data-testid="stMain"]'),
                doc.querySelector('[data-testid="stMainBlockContainer"]'),
                doc.querySelector('section.main'),
                doc.querySelector('main')
            ].filter(Boolean);

            const all = Array.from(doc.querySelectorAll('div, section, main'));
            all.forEach(el => {{
                if (el.scrollHeight > el.clientHeight + 80) candidates.push(el);
            }});

            return [...new Set(candidates)];
        }}

        function goToResult() {{
            const doc = window.parent.document;
            const win = window.parent;

            let el =
                doc.getElementById("card-resultado-avaliacao") ||
                doc.querySelector("[data-result-anchor='true']");

            if (!el) {{
                const all = Array.from(doc.querySelectorAll("div, span, h1, h2, h3, p"));
                el = all.find(x => (x.innerText || "").trim().includes("Resultado final da avaliação"));
            }}

            if (!el) return false;

            try {{
                el.scrollIntoView({{ behavior: "smooth", block: "start", inline: "nearest" }});
            }} catch (e) {{}}

            try {{
                const rect = el.getBoundingClientRect();
                const currentTop = win.pageYOffset || doc.documentElement.scrollTop || doc.body.scrollTop || 0;
                win.scrollTo({{ top: rect.top + currentTop - 30, behavior: "smooth" }});
            }} catch (e) {{}}

            try {{
                findScrollableParents(doc).forEach(scroller => {{
                    if (scroller && scroller.scrollHeight > scroller.clientHeight) {{
                        const rect = el.getBoundingClientRect();
                        const containerRect = scroller.getBoundingClientRect ? scroller.getBoundingClientRect() : {{ top: 0 }};
                        const currentTop = scroller.scrollTop || 0;
                        const targetTop = currentTop + rect.top - containerRect.top - 30;
                        scroller.scrollTo({{ top: targetTop, behavior: "smooth" }});
                    }}
                }});
            }} catch (e) {{}}

            return true;
        }}

        setTimeout(goToResult, 50);
        setTimeout(goToResult, 250);
        setTimeout(goToResult, 600);
        setTimeout(goToResult, 1000);
        setTimeout(goToResult, 1600);
        setTimeout(goToResult, 2300);
        </script>
        """, height=0)

    st.session_state.scroll_to_result = False
    st.session_state.force_result_scroll_after_extra = False


def run_scroll_to_bottom_script():
    if not st.session_state.get("scroll_to_bottom"):
        return

    nonce = st.session_state.get("scroll_nonce", 0)

    components.html(f"""
        <script>
        // força execução nova do scroll bottom: {nonce}

        function findScrollableParents(doc) {{
            const candidates = [
                doc.scrollingElement,
                doc.documentElement,
                doc.body,
                doc.querySelector('.stApp'),
                doc.querySelector('[data-testid="stAppViewContainer"]'),
                doc.querySelector('[data-testid="stMain"]'),
                doc.querySelector('[data-testid="stMainBlockContainer"]'),
                doc.querySelector('section.main'),
                doc.querySelector('main')
            ].filter(Boolean);

            const all = Array.from(doc.querySelectorAll('div, section, main'));
            all.forEach(el => {{
                if (el.scrollHeight > el.clientHeight + 80) candidates.push(el);
            }});

            return [...new Set(candidates)];
        }}

        function goBottom() {{
            const doc = window.parent.document;
            const win = window.parent;
            const bottomAnchor = doc.getElementById('fim-da-pagina');

            try {{
                if (bottomAnchor) {{
                    bottomAnchor.scrollIntoView({{ behavior: 'smooth', block: 'end', inline: 'nearest' }});
                }}
            }} catch(e) {{}}

            try {{
                win.scrollTo({{ top: doc.body.scrollHeight + 5000, behavior: 'smooth' }});
            }} catch(e) {{}}

            try {{
                findScrollableParents(doc).forEach(scroller => {{
                    if (scroller && scroller.scrollHeight > scroller.clientHeight) {{
                        scroller.scrollTo({{ top: scroller.scrollHeight + 5000, behavior: 'smooth' }});
                    }}
                }});
            }} catch(e) {{}}

            return true;
        }}

        setTimeout(goBottom, 50);
        setTimeout(goBottom, 200);
        setTimeout(goBottom, 450);
        setTimeout(goBottom, 800);
        setTimeout(goBottom, 1200);
        setTimeout(goBottom, 1800);
        setTimeout(goBottom, 2500);
        </script>
    """, height=0)
    st.session_state.scroll_to_bottom = False


# =============================
# AVALIAÇÃO
# =============================
def evaluate(question, reference, user_answer):
    llm = get_llm()

    prompt = f"""
Você é um avaliador de conhecimento em inteligência artificial.

Avalie a resposta do usuário com base na resposta esperada.

Regras:
- Considere correta se estiver conceitualmente certa, mesmo com palavras diferentes.
- Se estiver incompleta demais, errada ou fugir do tema, considere incorreta.
- Retorne obrigatoriamente neste formato exato:

correct: true ou false
score: 1 ou 0
feedback: texto curto em português

Pergunta: {question}
Resposta esperada: {reference}
Resposta do usuário: {user_answer}
"""

    res = llm.invoke(prompt)
    content = res.content.lower() if res.content else ""

    correct = "correct: true" in content
    score = 1 if correct else 0

    feedback = res.content if res.content else "Sem feedback gerado."
    if "feedback:" in content:
        parts = res.content.split("feedback:", 1)
        if len(parts) > 1:
            feedback = parts[1].strip()

    return AnswerEvaluation(correct=correct, score=score, feedback=feedback)



def get_question_intent(question: str) -> str:
    """Identifica a intenção da pergunta para melhorar a recuperação no RAG."""
    q = normalize_text(question)

    if any(term in q for term in ["endereco", "endereço", "onde fica", "localizacao", "localização", "localizada", "localizado", "cep"]):
        return "endereco"

    if any(term in q for term in ["telefone", "contato", "ligar", "numero", "número"]):
        return "telefone"

    if any(term in q for term in ["primeiro nome", "nome antigo", "nome inicial", "chamava", "abrigo"]):
        return "primeiro_nome"

    if any(term in q for term in ["fundacao", "fundação", "fundada", "fundado", "existencia", "existência", "anos"]):
        return "fundacao"

    return "geral"


def intent_terms(intent: str):
    """Termos fortes por intenção para aumentar a precisão do RAG."""
    mapping = {
        "endereco": ["endereço", "endereco", "localizada", "localizado", "avenida", "rua", "cep", "bairro"],
        "telefone": ["telefone", "contato", "informações", "informacoes"],
        "primeiro_nome": ["primeiro nome", "hospital de abrigo", "abrigo"],
        "fundacao": ["fundação", "fundacao", "fundada", "fundado", "existência", "existencia", "anos"],
        "geral": [],
    }
    return mapping.get(intent, [])


def retrieve_context_blindado(question: str, top_k: int = RAG_TOP_K):
    """
    Recuperação blindada:
    - usa a busca lexical original;
    - reforça termos de intenção como endereço, telefone e primeiro nome;
    - para perguntas internas da Santa Casa, tenta garantir um trecho da base antes de cair na OpenAI.
    """
    docs = build_rag_index()
    if not docs:
        return "", []

    q_norm = normalize_text(question)
    q_tokens = set(tokenize(question))
    intent = get_question_intent(question)
    strong_terms = intent_terms(intent)

    scored = []
    for doc in docs:
        text = doc.get("text", "")
        text_norm = normalize_text(text)
        doc_tokens = set(doc.get("tokens", []))

        score = 0.0

        # Pontuação por interseção de palavras relevantes.
        score += len(q_tokens.intersection(doc_tokens)) * 2

        # Pontuação por termos fortes da intenção.
        for term in strong_terms:
            if normalize_text(term) in text_norm:
                score += 8

        # Pontuação extra quando a pergunta menciona Santa Casa/Hospital e o trecho fala da instituição.
        if is_internal_santa_casa_question(question):
            if any(t in text_norm for t in ["santa casa", "hospital", "santa efigenia", "santa efigênia"]):
                score += 5

        # Bônus para correspondência da pergunta inteira.
        if q_norm and q_norm in text_norm:
            score += 10

        if score > 0:
            scored.append((score, doc))

    # Se a busca blindada não pontuar, tenta a busca original.
    if not scored:
        return retrieve_context(question, top_k=top_k)

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [doc for _, doc in scored[:top_k]]

    context_parts = []
    sources = []
    used_labels = set()
    for doc in selected:
        label = f"RAG_SantaCasaBH - trecho {doc['chunk_id']}"
        if label in used_labels:
            continue
        used_labels.add(label)
        sources.append(label)
        context_parts.append(f"[Fonte interna: {label}]\n{doc['text']}")

    return "\n\n---\n\n".join(context_parts), unique_list(sources)



def read_all_knowledge_text() -> str:
    """Lê todo o conteúdo bruto dos arquivos da base interna."""
    parts = []
    for path in get_knowledge_paths():
        try:
            text = read_text_file(path)
            if text:
                parts.append(text)
        except Exception:
            continue
    return "\n\n".join(parts).strip()


def clean_section_text(text: str) -> str:
    text = re.sub(r"#+", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def direct_answer_from_raw_knowledge(question: str):
    """
    Resposta blindada por dado objetivo usando o arquivo bruto.
    Isso evita o ranking pegar um trecho errado ou a OpenAI trocar dado interno.
    Retorna (resposta, fonte) ou ("", "").
    """
    raw = read_all_knowledge_text()
    if not raw:
        return "", ""

    intent = get_question_intent(question)
    raw_clean = clean_section_text(raw)

    if intent == "endereco" and is_internal_santa_casa_question(question):
        m = re.search(
            r"(?:ENDERE[ÇC]O\s+SANTA\s+CASA\s+BH\s*)?(A\s+Santa\s+Casa\s+BH\s+est[áa]\s+localizada\s+.*?(?:CEP\s*[\d\.\-]+))",
            raw_clean,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).strip(), "RAG_SantaCasaBH - trecho 1"

        m = re.search(r"(Avenida\s+[^.]+(?:CEP\s*[\d\.\-]+)?)", raw_clean, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(), "RAG_SantaCasaBH - trecho 1"

    if intent == "telefone" and is_internal_santa_casa_question(question):
        m = re.search(r"(\(?\d{2}\)?\s*\d{4,5}[-\s]?\d{4})", raw_clean, flags=re.IGNORECASE)
        if m:
            telefone = m.group(1).strip()
            return f"O telefone geral para contato e informações é {telefone}.", "RAG_SantaCasaBH - trecho 1"

    if intent == "primeiro_nome" and is_internal_santa_casa_question(question):
        m = re.search(r"(O\s+primeiro\s+nome\s+do\s+hospital\s+Santa\s+Casa\s+BH,?\s+foi\s+hospital\s+de\s+Abrigo)", raw_clean, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip() + ".", "RAG_SantaCasaBH - trecho 1"

    return "", ""

def extract_direct_fact_from_rag(question: str, rag_context: str) -> str:
    """
    Para dados objetivos da Santa Casa BH, responde diretamente com o texto do RAG.
    Isso evita a OpenAI trocar endereço/telefone por conhecimento externo.
    """
    if not question or not rag_context:
        return ""

    intent = get_question_intent(question)
    text = re.sub(r"\[Fonte interna:.*?\]", " ", rag_context)
    text = re.sub(r"\s+", " ", text).strip()

    # Quebra em frases mantendo informação compacta.
    sentences = re.split(r"(?<=[.!?])\s+|\s+###", text)
    sentences = [s.strip(" -#") for s in sentences if s and len(s.strip()) > 5]

    def find_sentence(terms):
        for sentence in sentences:
            s_norm = normalize_text(sentence)
            if any(normalize_text(term) in s_norm for term in terms):
                return sentence.strip()
        return ""

    if intent == "endereco":
        sentence = find_sentence(["localizada", "endereço", "endereco", "avenida", "rua", "cep"])
        if sentence:
            return sentence

    if intent == "telefone":
        sentence = find_sentence(["telefone", "contato", "informações", "informacoes"])
        if sentence:
            return sentence

    if intent == "primeiro_nome":
        sentence = find_sentence(["primeiro nome", "hospital de abrigo", "abrigo"])
        if sentence:
            return sentence

    if intent == "fundacao":
        sentence = find_sentence(["fundação", "fundacao", "fundada", "fundado", "existência", "existencia", "anos"])
        if sentence:
            return sentence

    return ""


def format_answer_with_source(answer: str, sources_text: str) -> str:
    """Remove fonte duplicada e adiciona uma fonte padronizada em linha separada."""
    answer = remove_fonte_lines(answer or "").strip()
    if not answer:
        answer = "Encontrei essa informação na base interna, mas não consegui montar uma resposta textual."
    return f"{answer}\n\nFonte: {sources_text}"


def is_internal_santa_casa_question(question: str) -> bool:
    """
    Só considera pergunta interna da Santa Casa quando a pergunta citar claramente a instituição.
    Não pode considerar 'endereço', 'telefone' ou 'cep' sozinho.
    """
    q = normalize_text(question)

    termos_santa_casa = [
        "santa casa",
        "santa casa bh",
        "hospital santa casa",
        "hospital santa casa bh",
        "francisco sales",
        "santa efigenia",
        "santa efigênia"
    ]

    return any(term in q for term in termos_santa_casa)

def is_rag_context_relevant(question: str, rag_context: str) -> bool:
    """
    Valida se o RAG deve ser usado.

    Regra principal:
    - Se a pergunta for sobre Santa Casa BH e o RAG encontrou algum trecho,
    usa o RAG obrigatoriamente para evitar resposta externa divergente.
    - Para perguntas gerais, usa uma validação lexical simples.
    """
    if not question or not rag_context:
        return False

    # Perguntas internas devem priorizar a base interna quando houver trecho recuperado.
    if is_internal_santa_casa_question(question):
        return True

    question_tokens = set(tokenize(question))
    context_tokens = set(tokenize(rag_context))

    if not question_tokens or not context_tokens:
        return False

    common_tokens = question_tokens.intersection(context_tokens)

    # Para perguntas curtas, 1 termo relevante já é suficiente.
    if len(question_tokens) <= 3:
        return len(common_tokens) >= 1

    # Para perguntas maiores, exige pelo menos 2 termos ou 30% de interseção.
    return len(common_tokens) >= 2 or (len(common_tokens) / max(len(question_tokens), 1)) >= 0.30

# =============================
# CHAT NORMAL / RAG SIMPLES
# =============================
def answer_free_chat(user_text: str) -> str:
    llm = get_llm()

    pergunta = user_text.strip()
    pergunta_norm = normalize_text(pergunta)

    # Detecta pergunta sobre Santa Casa, inclusive quando usa "ela"
    pergunta_santa_casa = is_internal_santa_casa_question(pergunta)

    if any(t in pergunta_norm for t in ["ela", "dela", "fundada", "fundou", "provedor", "historia", "história"]):
        pergunta_santa_casa = True

    # 1) Se for pergunta da Santa Casa, busca primeiro no RAG
    if pergunta_santa_casa:
        rag_context, rag_sources = retrieve_context_blindado(pergunta)
        tem_rag = bool(rag_context and rag_sources)

        direct_fact_raw, source_raw = direct_answer_from_raw_knowledge(pergunta)
        if direct_fact_raw:
            return format_answer_with_source(direct_fact_raw, source_raw)

        if tem_rag:
            sources_text = ", ".join(unique_list(rag_sources))

            prompt_rag = f"""
Você é a IA For HEALTH da Santa Casa BH.

Responda SOMENTE com base no CONTEXTO abaixo.

REGRAS:
- Não use conhecimento externo.
- Não invente informação.
- Não copie títulos como "FAQ DIRETA".
- Não repita a pergunta.
- Responda de forma curta, direta e natural.
- Se a resposta estiver no contexto, responda apenas a resposta final.
- Se não estiver no contexto, responda exatamente:
Essa informação não foi localizada na base interna da Santa Casa BH.

CONTEXTO:
{rag_context}

PERGUNTA:
{pergunta}

RESPOSTA FINAL:
"""

            res = llm.invoke(prompt_rag)
            resposta = res.content.strip() if res.content else ""

            if resposta and "não foi localizada" not in resposta.lower():
                return format_answer_with_source(resposta, sources_text)

        return (
            "Essa informação não foi localizada na base interna da Santa Casa BH.\n\n"
            "Para responder corretamente, atualize o arquivo rag_santa_casa_bh_2024.txt com essa informação."
        )

    # 2) Se NÃO for Santa Casa, responde pela OpenAI normalmente
    prompt = f"""
Você é a IA For HEALTH, um assistente profissional e objetivo.

Responda em português do Brasil.

Regras:
- Se a pergunta for sobre inteligência artificial, responda normalmente.
- Seja claro, educado e direto.
- Não diga que consultou RAG.
- Não invente que a informação está na base interna.

Pergunta:
{pergunta}
"""

    res = llm.invoke(prompt)
    return res.content.strip() if res.content else "Não consegui responder agora."

# =============================
# PROCESSAMENTO
# =============================

def contem_palavra_ofensiva(texto: str) -> bool:
    texto_norm = normalize_text(texto)

    palavras_bloqueadas = [
        "puta",
        "fdp",
        "burro",
        "cu"
    ]

    return any(p in texto_norm.split() for p in palavras_bloqueadas)


def nome_valido(nome: str) -> bool:
    nome = nome.strip()

    if len(nome) < 3:
        return False

    if contem_palavra_ofensiva(nome):
        return False

    # Aceita letras, espaços e acentos
    if not re.match(r"^[A-Za-zÀ-ÿ\s]+$", nome):
        return False

    return True


def matricula_valida(matricula: str) -> bool:
    matricula = matricula.strip()

    # Somente números
    if not matricula.isdigit():
        return False

    # Ajuste aqui a quantidade mínima/máxima conforme sua regra
    if len(matricula) < 3 or len(matricula) > 6:
        return False

    return True

def validar_nome_com_openai(nome: str) -> tuple[bool, str]:
    llm = get_llm()

    prompt = f"""
Você é um validador de cadastro.

Analise se o texto abaixo parece ser um nome próprio real de pessoa.

Texto informado:
{nome}

Regras:
- Bloqueie palavrões, ofensas, xingamentos ou termos inadequados.
- Bloqueie frases aleatórias, brincadeiras, letras repetidas sem sentido ou nomes muito estranhos.
- Aceite nomes reais, inclusive nomes compostos e com acentos.
- Responda somente neste formato:

valido: sim ou nao
motivo: texto curto
"""

    res = llm.invoke(prompt)
    resposta = res.content.strip().lower() if res.content else ""

    valido = "valido: sim" in resposta or "válido: sim" in resposta

    motivo = "Nome inválido."
    if "motivo:" in resposta:
        motivo = resposta.split("motivo:", 1)[1].strip()

    return valido, motivo


def process_name(user_text: str):
    add_msg("user", user_text, section="quiz")

    if not nome_valido(user_text):
        add_msg(
            "assistant",
            "Nome inválido. Informe seu nome corretamente, sem números, símbolos ou palavras ofensivas.",
            section="quiz"
        )
        st.session_state.focus_input = True
        st.session_state.scroll_to_bottom = True
        st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1
        return

    valido_openai, motivo = validar_nome_com_openai(user_text)

    if not valido_openai:
        add_msg(
            "assistant",
            f"Nome não aceito. {motivo} Informe um nome válido para continuar.",
            section="quiz"
        )
        st.session_state.focus_input = True
        st.session_state.scroll_to_bottom = True
        st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1
        return

    st.session_state.name = user_text.strip()
    st.session_state.phase = "matricula"
    st.session_state.focus_input = True
    st.session_state.scroll_to_bottom = True
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1
    add_msg("assistant", f"Prazer, {st.session_state.name}! Qual é a sua matrícula?", section="quiz")


def process_matricula(user_text: str):
    if not matricula_valida(user_text):
        add_msg("user", user_text, section="quiz")
        add_msg(
            "assistant",
            "Matrícula inválida. Informe apenas números, sem letras, espaços ou símbolos.",
            section="quiz"
        )
        st.session_state.focus_input = True
        st.session_state.scroll_to_bottom = True
        st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1
        return

    add_msg("user", user_text, section="quiz")
    st.session_state.matricula = user_text.strip()
    st.session_state.phase = "unidade"
    st.session_state.focus_input = False
    st.session_state.scroll_to_bottom = True
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1
    add_msg("assistant", "Qual é a sua unidade de negócio? Selecione uma das opções abaixo:", section="quiz")

TEMAS_COMPARTILHAMENTO = [
    "Fundamentos de IA",
    "Machine Learning",
    "Deep Learning",
    "Processamento de Linguagem Natural (NLP)",
    "IA Generativa",
    "Large Language Models (LLMs)",
    "Engenharia de Prompt",
    "RAG (Retrieval-Augmented Generation)",
    "Fine-Tuning de Modelos",
    "Embeddings e Busca Vetorial",
    "Agentes de IA (AI Agents)",
    "Speech AI e IA de Voz",
    "IA Multimodal",
    "Automação Inteligente",
    "IA Aplicada à Saúde",
    "IA Aplicada a Negócios",
    "IA para Produtividade",
    "IA para Experiência do Usuário",
    "Governança de IA",
    "Ética em IA",
    "Segurança em IA",
    "Compliance e LGPD em IA",
    "Infraestrutura para IA",
    "APIs e Integrações de IA",
    "Cloud AI",
    "Sistemas Autônomos",
    "Analytics Preditivo",
    "IA Conversacional",
    "Sistemas Especialistas",
    "Transformação Digital com IA",
    "IA e Futuro do Trabalho",
    "IA e Educação",
    "IA e Pesquisa Científica",
    "IA e Cibersegurança",
    "IA e IoT"
]

def process_unidade(unidade: str):
    if unidade not in UNIDADES_NEGOCIO:
        add_msg(
            "assistant",
            "Unidade inválida. Selecione uma das opções disponíveis.",
            section="quiz"
        )
        return
    
    st.session_state.unidade_negocio = unidade

    add_msg("user", unidade, section="quiz")
    st.session_state.phase = "confirmar_quiz"
    st.session_state.focus_input = False
    st.session_state.scroll_to_bottom = True
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1

    add_msg(
    "assistant",
    "Deseja responder às perguntas sobre Inteligência Artificial para que seja validado o seu nível de conhecimento? Caso não queira participar, pode responder não. Seu nível será Básico.",
    section="quiz"
)
    
    
def process_quiz_answer(user_text: str):
    add_msg("user", user_text, section="quiz")

    current = st.session_state.questions[st.session_state.index]
    result = evaluate(current["question"], current["reference"], user_text.strip())

    st.session_state.results.append({
        "question": current["question"],
        "reference": current["reference"],
        "user_answer": user_text.strip(),
        "correct": result.correct,
        "score": result.score,
        "feedback": result.feedback,
    })

    st.session_state.index += 1

    # IMPORTANTE: força scroll para o fim também na primeira pergunta e nas demais.
    st.session_state.scroll_to_bottom = True
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1

    if st.session_state.index < 5:
        next_question = st.session_state.questions[st.session_state.index]["question"]
        st.session_state.phase = "quiz"
        st.session_state.focus_input = True
        add_msg("assistant", next_question, section="quiz")
        return

    total = sum(item["score"] for item in st.session_state.results)
    st.session_state.score = total
    st.session_state.level = classify(total)
    st.session_state.quiz_completed = True
    st.session_state.scroll_to_bottom = True
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1

    add_msg(
        "assistant",
        f"{st.session_state.name}, sua avaliação foi concluída. Confira abaixo seu resultado final e a revisão das perguntas.",
        section="quiz"
    )

    if st.session_state.level in ["Intermediário", "Avançado"]:
        st.session_state.extra_question_needed = True
        st.session_state.extra_question_answered = False
        st.session_state.extra_question_processing = False
        st.session_state.phase = "mostrar_resultado"
    else:
        st.session_state.extra_question_needed = False
        st.session_state.extra_question_answered = True
        st.session_state.phase = "chat"
        st.session_state.focus_input = True
        st.session_state.test_end_time = datetime.now()
        save_result_to_file()


def process_free_chat(user_text: str):
    add_msg("user", user_text, section="post_result")
    answer = answer_free_chat(user_text)
    add_msg("assistant", answer, section="post_result")
    st.session_state.focus_input = True
    st.session_state.scroll_to_bottom = True
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1


def process_user_message(user_text: str):
    if not user_text or not user_text.strip():
        return

    if st.session_state.phase == "name":
        process_name(user_text)
        return

    if st.session_state.phase == "matricula":
        process_matricula(user_text)
        return

    if st.session_state.phase == "unidade":
        process_unidade(user_text)
        return

    if st.session_state.phase == "quiz":
        process_quiz_answer(user_text)
        return

    if st.session_state.phase == "tema_compartilhamento":
        add_msg(
            "assistant",
            "Selecione um ou mais temas nas opções disponíveis e clique em Salvar temas.",
            section="quiz"
        )

        st.session_state.scroll_to_bottom = True
        return

    if st.session_state.phase == "chat":
        process_free_chat(user_text)
        return


# =============================
# UI
# =============================
def render_header():
    st.markdown(f"""
    <div class="top-header">
    <div></div>

    <div>
        <h1 class="app-title-center">{APP_TITLE}</h1>
    </div>

    <div class="app-version-right">
        Versão da IA: {APP_VERSION}
    </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="start-card-shell">', unsafe_allow_html=True)

        col_texto, col_botao = st.columns([5, 1.2])

        with col_texto:
            st.markdown("""
            <div class="start-card-text">
            <div class="start-card-title">Pronto para começar?</div>
            <div class="start-card-subtitle">
                Inicie sua avaliação de conhecimento em Inteligência Artificial.
            </div>
            </div>
            """, unsafe_allow_html=True)

        with col_botao:
            if not st.session_state.started:
                st.markdown('<div class="btn-primary-wrap">', unsafe_allow_html=True)
                if st.button("Começar avaliação", key="btn_iniciar_conversa", use_container_width=True):
                    start_conversation()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.markdown('<div class="btn-secondary-wrap">', unsafe_allow_html=True)
                if st.button("Reiniciar avaliação", key="btn_reiniciar_conversa", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()


def render_status():
    if not st.session_state.started:
        return

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Nome", st.session_state.name if st.session_state.name else "-")
    with col2:
        st.metric("Matrícula", st.session_state.matricula if st.session_state.matricula else "-")
    with col3:
        st.metric("Unidade", st.session_state.unidade_negocio if st.session_state.unidade_negocio else "-")
    with col4:
        score_text = f"{st.session_state.score}/5" if st.session_state.quiz_completed else "-"
        st.metric("Pontuação", score_text)
    with col5:
        st.metric("Nível", st.session_state.level if st.session_state.quiz_completed else "-")

    st.divider()


def render_chat_block(messages):
    for m in messages:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                st.markdown("**IA For HEALTH:**")
            else:
                st.markdown("**Você digitou:**")
            texto_mensagem = str(m.get("content", ""))
            st.markdown(texto_mensagem.replace("\n", "  \n"))


def render_extra_question_buttons():
    if not (
        st.session_state.quiz_completed
        and st.session_state.phase == "mostrar_resultado"
        and st.session_state.extra_question_needed
        and not st.session_state.extra_question_answered
    ):
        return

    st.markdown(
        '<div class="result-section-title">Deseja compartilhar seus conhecimentos sobre Inteligência Artificial?</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1, 6])

    with col1:
        if st.button("✅ Sim", key="btn_extra_sim"):
            responder_pergunta_extra("Sim")

    with col2:
        if st.button("❌ Não", key="btn_extra_nao"):
            responder_pergunta_extra("Não")

def render_unidade_buttons():
    if not (st.session_state.started and st.session_state.phase == "unidade"):
        return

    st.markdown(
        '<div class="result-section-title">Selecione sua unidade de negócio</div>',
        unsafe_allow_html=True
    )

    # largura menor e alinhado à esquerda
    col1, col2 = st.columns([2, 6])

    with col1:
        if st.button("🏥 Santa Casa BH", key="btn_unidade_santa_casa"):
            process_unidade("Santa Casa BH")
            st.rerun()

        if st.button("🏥 São Lucas", key="btn_unidade_sao_lucas"):
            process_unidade("São Lucas")
            st.rerun()

        if st.button("🧩 Centro de Autismo", key="btn_unidade_autismo"):
            process_unidade("Centro de Autismo")
            st.rerun()

        if st.button("⚰️ Funerária e Assistência Familiar", key="btn_unidade_funeraria"):
            process_unidade("Funerária e Assistência Familiar")
            st.rerun()

        if st.button("🎓 Faculdade de Saúde", key="btn_unidade_faculdade"):
            process_unidade("Faculdade de Saúde")
            st.rerun()

        if st.button("🩺 Ambulatórios Especializados", key="btn_unidade_ambulatorios"):
            process_unidade("Ambulatórios Especializados")
            st.rerun()

        if st.button("👴 Instituto Geriátrico", key="btn_unidade_geriatrico"):
            process_unidade("Instituto Geriátrico")
            st.rerun()

        if st.button("👶 Instituto Materno Infantil", key="btn_unidade_materno"):
            process_unidade("Instituto Materno Infantil")
            st.rerun()

        if st.button("🎗️ Instituto de Oncologia", key="btn_unidade_oncologia"):
            process_unidade("Instituto de Oncologia")
            st.rerun()

        if st.button("🔬 Pesquisa Clínica", key="btn_unidade_pesquisa"):
            process_unidade("Pesquisa Clínica")
            st.rerun()

        if st.button("🧪 Órix Lab", key="btn_unidade_orix"):
            process_unidade("Órix Lab")
            st.rerun()

        if st.button("🏢 Corporativo", key="btn_unidade_corporativo"):
            process_unidade("Corporativo")
            st.rerun()
            
def render_confirmar_quiz_buttons():
    if not (st.session_state.started and st.session_state.phase == "confirmar_quiz"):
        return

    st.markdown(
        '<div class="result-section-title">Deseja participar da avaliação?</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1, 6])

    with col1:
        if st.button("✅ Sim", key="btn_confirmar_quiz_sim"):
            st.session_state.aceitou_responder_quiz = "Sim"
            add_msg("user", "Sim", section="quiz")

            st.session_state.questions = random.sample(QUESTION_BANK, 5)
            st.session_state.index = 0
            st.session_state.results = []
            st.session_state.phase = "quiz"
            st.session_state.focus_input = True
            st.session_state.scroll_to_bottom = True
            st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1

            add_msg("assistant", st.session_state.questions[0]["question"], section="quiz")
            st.rerun()

    with col2:
        if st.button("❌ Não", key="btn_confirmar_quiz_nao"):
            st.session_state.aceitou_responder_quiz = "Não"
            add_msg("user", "Não", section="quiz")

            st.session_state.score = 0
            st.session_state.level = "Básico"
            st.session_state.quiz_completed = True
            st.session_state.extra_question_needed = False
            st.session_state.extra_question_answered = True
            st.session_state.phase = "chat"
            st.session_state.focus_input = True
            st.session_state.test_end_time = datetime.now()
            st.session_state.scroll_to_bottom = True
            st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1

            add_msg(
                "assistant",
                f"{st.session_state.name}, sua participação foi registrada como não participante da avaliação. Seu nível foi definido como Básico.",
                section="quiz"
            )

            save_result_to_file()
            st.rerun()

def render_final_result():
    if not st.session_state.quiz_completed:
        return

    st.markdown(
        '<div id="card-resultado-avaliacao" data-result-anchor="true" class="result-section-title" style="display:block;">Resultado final da avaliação</div>',
        unsafe_allow_html=True
    )

    extra_info_html = ""

    if st.session_state.extra_question_answered and st.session_state.extra_question_answer:
        extra_info_html += f"""
<div class="final-result-info-item final-result-info-item-full">
    <span class="final-result-info-label">Interesse em compartilhar conhecimento?</span>
    <span class="final-result-info-value">{st.session_state.extra_question_answer}</span>
</div>
"""

    if st.session_state.extra_question_answer == "Sim" and st.session_state.extra_question_topic:
        extra_info_html += f"""
<div class="final-result-info-item final-result-info-item-full">
    <span class="final-result-info-label">Tema que deseja abordar no ensinamento</span>
    <span class="final-result-info-value">{st.session_state.extra_question_topic}</span>
</div>
"""

    level_class = get_level_class(st.session_state.level)

    st.markdown(
f"""<div class="final-result-card">
<div class="final-result-top">
    <div>
        <div class="final-result-name">{st.session_state.name}</div>
        <div class="final-result-subtitle">Resultado do diagnóstico de conhecimento em IA</div>
    </div>
    <div class="final-result-score-box">
        <div class="final-result-score-label">Pontuação</div>
        <div class="final-result-score-value">{st.session_state.score}/5</div>
    </div>
</div>

<div class="final-result-level-row">
    <span class="final-result-level-label">Nível de conhecimento identificado:</span>
    <span class="final-result-level {level_class}">{st.session_state.level}</span>
</div>

<div class="final-result-info-grid">
    <div class="final-result-info-item">
        <span class="final-result-info-label">Matrícula</span>
        <span class="final-result-info-value">{st.session_state.matricula}</span>
    </div>
    <div class="final-result-info-item">
        <span class="final-result-info-label">Unidade de negócio</span>
        <span class="final-result-info-value">{st.session_state.unidade_negocio}</span>
    </div>
    <div class="final-result-info-item">
        <span class="final-result-info-label">Início</span>
        <span class="final-result-info-value">{format_datetime(st.session_state.test_start_time)}</span>
    </div>
    <div class="final-result-info-item">
        <span class="final-result-info-label">Versão do chat</span>
        <span class="final-result-info-value">{APP_VERSION}</span>
    </div>
    {extra_info_html}
</div>
</div>""",
        unsafe_allow_html=True
    )

    # Botão reiniciar dentro do card final, logo após o resumo visual do resultado.
    with st.container():
        if st.button("🔄 Reiniciar teste", use_container_width=True, key="btn_reiniciar_dentro_card"):
            reiniciar_teste()

    st.markdown(
        '<div class="result-section-title">Revisão das perguntas</div>',
        unsafe_allow_html=True
    )

    for item in st.session_state.results:
        status_class = "result-ok" if item["correct"] else "result-error"
        status_text = "Correta" if item["correct"] else "Incorreta"

        st.markdown(
f"""<div class="question-card">
<div class="question-status {status_class}">{status_text}</div>
<div class="question-title">{item['question']}</div>
<div class="question-block">
<div class="question-label">Sua resposta</div>
<div class="question-text">{item['user_answer']}</div>
</div>
<div class="question-block">
<div class="question-label">Resposta correta</div>
<div class="question-text">{item['reference']}</div>
</div>
<div class="question-block">
<div class="question-label">Comentário</div>
<div class="question-text">{item['feedback']}</div>
</div>
</div>""",
            unsafe_allow_html=True
        )

    
def render_chat_actions():
    if not st.session_state.quiz_completed:
        return

    st.markdown('<div class="chat-actions-wrap">', unsafe_allow_html=True)

    if st.button("🔙 Ver resultado da avaliação", use_container_width=True, key="btn_ver_resultado_final"):
        voltar_resultado()

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.phase == "chat":
        st.markdown(
            '''
            <div class="chat-location-card">
                🤖 <strong>Chat para aperfeiçoamento</strong><br>
                O chat fica logo abaixo. Use o campo de mensagem para tirar dúvidas sobre 
                inteligência artificial e consultar conteúdos.
            </div>
            ''',
            unsafe_allow_html=True
        )

def render_card_explicacao_chat():
    st.markdown(
        '''
        <div class="home-grid">

        <div class="home-card">
            🤖 <strong>Sobre a IA For Health</strong><br>
            A <strong>IA For Health</strong> apoia o desenvolvimento dos colaboradores da Santa Casa BH no tema de Inteligência Artificial.
            A ferramenta identifica o nível de conhecimento e direciona uma experiência personalizada de aprendizado.
        </div>

        <div class="home-card">
            💡 <strong>Como funciona</strong>

        <div class="como-funciona-grid compacto">
        <div class="como-item">
            <strong>1. Identificação inicial</strong><br>
            Dados básicos para contextualizar sua experiência.<br><br>
            <strong>2. Avaliação</strong><br>
            Perguntas sobre conceitos de IA.<br><br>
            <strong>3. Análise</strong><br>
            Identificação do nível com base nas respostas.
        </div>
        

        <div class="como-item">
            <strong>4. Aprendizado</strong><br>
            Chat para dúvidas, conceitos e exemplos práticos.<br>
             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Tirar dúvidas<br>
             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Explorar conceitos<br>
             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Ver exemplos aplicados à saúde<br>
             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Acessar conteúdos internos da instituição<br>
        </div>
        </div>
        </div>

        </div>
        ''',
        unsafe_allow_html=True
    )

def render_explicacao_nivel_final():
    nivel = st.session_state.level

    if nivel == "Básico":
        card_nivel = """
<div class="next-step-card">
    🔴 <strong>Nível 1: Básico (Alfabetização em IA)</strong><br>
    Neste estágio, o colaborador é um consumidor consciente. Ele entende o que é a IA e como ela pode auxiliá-lo em tarefas simples.<br><br>
    <strong>Conceito:</strong> Capacidade de utilizar ferramentas prontas para otimizar a rotina.<br>
    <strong>Exemplo Prático:</strong> Usar o ChatGPT para redigir um e-mail, resumir um documento longo ou pedir ideias de pautas para uma reunião. O foco é ganhar agilidade operacional.
</div>
"""
    elif nivel == "Intermediário":
        card_nivel = """
<div class="next-step-card">
    🟡 <strong>Nível 2: Intermediário (Uso Estratégico)</strong><br>
    Aqui, o colaborador atua como um otimizador de processos. Ele não apenas usa a ferramenta, mas sabe como extrair o melhor dela através de técnicas mais refinadas.<br><br>
    <strong>Conceito:</strong> Domínio de engenharia de prompt e integração da IA em fluxos de trabalho existentes.<br>
    <strong>Exemplo Prático:</strong> Criar prompts complexos que incluam contexto, tom de voz e formato específico. Utilizar IA para análise de dados básica em planilhas ou para criar apresentações estruturadas. O foco é a qualidade e personalização.
</div>
"""
    else:
        card_nivel = """
<div class="next-step-card">
    🟢 <strong>Nível 3: Avançado (Inovação e Automação)</strong><br>
    O colaborador avançado é um arquiteto de soluções. Ele tem visão sistêmica e consegue conectar diferentes tecnologias para criar soluções automáticas ou preditivas.<br><br>
    <strong>Conceito:</strong> Capacidade de configurar automações, realizar análises preditivas e entender as implicações éticas e de governança.<br>
    <strong>Exemplo Prático:</strong> Configurar um fluxo que lê e-mails automaticamente, extrai dados relevantes e os insere em um CRM usando IA. Ou usar modelos para prever tendências de mercado com base em grandes volumes de dados. O foco é a transformação de processos.
</div>
"""

    st.markdown(
        card_nivel +
        """
<div class="next-step-card">
    <strong>Resumindo</strong><br>
    <strong>Básico:</strong> Sabe o que é e usa para tarefas simples (resumos, e-mails).<br>
    <strong>Intermediário:</strong> Domina a técnica para resultados específicos e complexos.<br>
    <strong>Avançado:</strong> Cria processos novos e automatiza tarefas de ponta a ponta.
</div>
""",
        unsafe_allow_html=True
    )

def render_resumo_resultado():
    if not st.session_state.quiz_completed:
        return

    level_class = get_level_class(st.session_state.level)

    st.markdown(f"""
<div class="final-result-card">

<div class="final-result-top">
<div>
<div class="final-result-name">{st.session_state.name}</div>
<div class="final-result-subtitle">Resumo da sua avaliação</div>
</div>

<div class="final-result-score-box">
<div class="final-result-score-label">Pontuação</div>
<div class="final-result-score-value">{st.session_state.score}/5</div>
</div>
</div>

<div class="final-result-level-row">
<span class="final-result-level-label">Nível de conhecimento:</span>
<span class="final-result-level {level_class}">
{st.session_state.level}
</span>
</div>

<!-- 🔥 ESSA PARTE DEIXA IGUAL AO CARD PRINCIPAL -->
<div class="final-result-info-grid">
<div class="final-result-info-item">
<span class="final-result-info-label">Nome</span>
<span class="final-result-info-value">{st.session_state.name}</span>
</div>

<div class="final-result-info-item">
<span class="final-result-info-label">Pontuação</span>
<span class="final-result-info-value">{st.session_state.score}/5</span>
</div>

<div class="final-result-info-item">
<span class="final-result-info-label">Nível</span>
<span class="final-result-info-value">{st.session_state.level}</span>
</div>
</div>

</div>
""", unsafe_allow_html=True)


def render_chat_messages():
    quiz_messages = [m for m in st.session_state.chat if m.get("section", "quiz") == "quiz"]
    post_result_messages = [m for m in st.session_state.chat if m.get("section") == "post_result"]

    if quiz_messages:
        render_chat_block(quiz_messages)
        render_unidade_buttons()    

    if st.session_state.quiz_completed:
        render_final_result()
        render_resumo_resultado()
        render_explicacao_nivel_final()
        

    render_extra_question_buttons()
    render_tema_compartilhamento_form()

    if st.session_state.quiz_completed:
        render_chat_actions()

    if st.session_state.phase == "chat":
        st.markdown('<div class="result-section-title">Chat para aperfeiçoamento</div>', unsafe_allow_html=True)

        if post_result_messages:
            render_chat_block(post_result_messages)        
    

    st.markdown('<div id="fim-da-pagina"></div>', unsafe_allow_html=True)

    if st.session_state.get("pending_scroll_bottom"):
        st.session_state.scroll_to_bottom = True
        st.session_state.pending_scroll_bottom = False


def bloquear_tradutor_google():
    components.html("""
        <script>
        const doc = window.parent.document;

        doc.documentElement.setAttribute("lang", "pt-BR");
        doc.documentElement.setAttribute("translate", "no");
        doc.body.setAttribute("translate", "no");
        doc.body.classList.add("notranslate");

        let meta = doc.querySelector('meta[name="google"]');
        if (!meta) {
            meta = doc.createElement("meta");
            meta.name = "google";
            meta.content = "notranslate";
            doc.head.appendChild(meta);
        }
        </script>
    """, height=0)

# =============================
# MAIN
# =============================
def main():
    st.set_page_config(layout="wide", page_title=APP_TITLE)

    st.markdown("""
    <style>
    [data-testid="stSidebar"] {display:none;}
    [data-testid="collapsedControl"] {display:none;}

    .stApp {
        background: #f7f9fc;
    }

    .result-section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 24px;
        margin-bottom: 12px;
        scroll-margin-top: 24px;
        letter-spacing: 0.2px;
    }

    .final-result-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #d6e4f0;
        border-left: 6px solid #1d4ed8;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 12px;
    }

    .final-result-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
        flex-wrap: wrap;
        padding-bottom: 12px;
        border-bottom: 1px solid #e5edf5;
    }

    .final-result-name {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
        margin-bottom: 3px;
    }

    .final-result-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }

    .final-result-score-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 14px;
        padding: 12px 16px;
        min-width: 130px;
        text-align: center;
    }

    .final-result-score-label {
        font-size: 0.75rem;
        color: #475569;
        font-weight: 700;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .final-result-score-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1d4ed8;
        line-height: 1.1;
    }

    .final-result-level-row {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 14px;
    }

    .final-result-level-label {
        font-size: 0.95rem;
        color: #334155;
        font-weight: 700;
        margin-right: 2px;
    }

    .final-result-level {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 800;
        line-height: 1;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.3);
    }

    .level-basico {
        background: #fff7ed;
        color: #c2410c;
        border: 1px solid #fdba74;
    }

    .level-intermediario {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #93c5fd;
    }

    .level-avancado {
        background: #ecfdf3;
        color: #166534;
        border: 1px solid #86efac;
    }

    .final-result-info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 12px;
    }

    .final-result-info-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }

    .final-result-info-item-full {
        grid-column: 1 / -1;
        background: #f8fafc;
    }

    .final-result-info-label {
        display: block;
        font-size: 0.74rem;
        font-weight: 800;
        color: #64748b;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.45px;
    }

    .final-result-info-value {
        display: block;
        font-size: 0.98rem;
        font-weight: 600;
        color: #0f172a;
        line-height: 1.35;
    }

    .question-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }

    .question-status {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .result-ok {
        background: #ecfdf3;
        color: #166534;
        border: 1px solid #bbf7d0;
    }

    .result-error {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    .question-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 16px;
    }

    .question-block {
        margin-bottom: 14px;
    }

    .question-label {
        font-size: 0.9rem;
        font-weight: 700;
        color: #475569;
        margin-bottom: 6px;
    }

    .question-text {
        font-size: 0.98rem;
        color: #334155;
        line-height: 1.55;
        background: #f8fafc;
        border-radius: 12px;
        padding: 12px 14px;
        border: 1px solid #e2e8f0;
    }

    .next-step-card {
        background: #eef6ff;
        border: 1px solid #dbeafe;
        color: #1e3a8a;
        border-radius: 18px;
        padding: 18px 20px;
        margin-top: 8px;
        margin-bottom: 10px;
        font-size: 1rem;
        line-height: 1.5;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.08);
    }

    .extra-btn-note {
        font-size: 0.92rem;
        color: #64748b;
        margin-top: 6px;
        margin-bottom: 10px;
    }

    .chat-actions-wrap {
        margin-top: 8px;
        margin-bottom: 16px;
    }

    .chat-location-card {
        background: #f8fafc;
        border: 1px dashed #93c5fd;
        color: #334155;
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 6px;
        margin-bottom: 16px;
        font-size: 0.95rem;
        line-height: 1.45;
    }

    div[data-testid="stChatInput"] {
        border: 1px solid #2563eb !important;
        border-radius: 16px !important;
        background: white !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border: 1px solid #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important;
    }

    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] input {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 16px !important;
    }

    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] > div > div {
        border: none !important;
        box-shadow: none !important;
        border-radius: 16px !important;
        background: white !important;
    }

    .stButton button {
        background: #2563eb !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
    }

    button[kind="secondary"] {
        border-radius: 12px !important;
    }
    
    .banner-topo-chat {
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        padding: 14px 18px;
        border-radius: 0 0 14px 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.25);
        font-size: 0.95rem;
        line-height: 1.4;
    }

    .banner-topo-chat strong {
        font-weight: 800;
    }

    .header-button-center {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }

    .header-button-center div[data-testid="stButton"] {
        width: 260px;
    }

    .footer-santa-ideia {
    position: fixed !important;
    bottom: 6px !important;
    left: 0 !important;
    width: 100% !important;
    text-align: center !important;
    color: #64748b !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    z-index: 999999 !important;
    pointer-events: none !important;
    }

    div[data-testid="stChatInput"] {
        margin-bottom: 34px !important;
    }
                
    [data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
    }

    .block-container {
        padding-top: 1rem !important;
    }

                    
    [data-testid="stMainBlockContainer"],
    .block-container {
        padding-top: 0.2rem !important;
    }

   .app-title-center {
    margin-top: 0px !important;
    margin-bottom: 4px !important;
    }

    .header-button-center {
        margin-top: 4px !important;
        margin-bottom: 4px !important;
    }

    [data-testid="stCaptionContainer"] {
        margin-top: -8px !important;
        margin-bottom: 4px !important;
    }

    hr {
        margin-top: 8px !important;
        margin-bottom: 12px !important;
    }    

    .header-button-center {
        display: flex !important;
        justify-content: center !important;
        margin-top: -8px !important;
        margin-bottom: 0px !important;
    }

    .header-button-center div[data-testid="stButton"] {
        width: 260px !important;
    }

    div[data-testid="stCaptionContainer"] {
        margin-top: -10px !important;
        margin-bottom: -10px !important;
        text-align: center !important;
    }

    hr {
        margin-top: 4px !important;
        margin-bottom: 8px !important;
    }

    .next-step-card:first-of-type {
        margin-top: 0px !important;
    }
                
    .como-funciona-grid {
    display: flex;
    gap: 20px;
    margin-top: 10px;
    }

    .como-col {
        flex: 1;
    }

    .como-item {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #1e293b;
        line-height: 1.4;
    }
                
   .home-grid {
    display: grid;
    grid-template-columns: 1fr 1.3fr;
    gap: 14px;
    margin-top: 10px;
    }

    .home-card {
        background: #eef6ff;
        border: 1px solid #d6e8ff;
        border-radius: 16px;
        padding: 16px 18px;
        color: #0f2f7a;
        font-size: 1rem;
        line-height: 1.45;
    }

    .como-funciona-grid.compacto {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 12px;
    }

    .como-item {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 12px 14px;
        color: #0f172a;
        font-size: 0.95rem;
        line-height: 1.35;
    }

    @media (max-width: 900px) {
        .home-grid {
            grid-template-columns: 1fr;
        }

        .como-funciona-grid.compacto {
            grid-template-columns: 1fr;
        }
    }
                
    .top-header {
    display: grid;
    grid-template-columns: 1fr 2fr 1fr;
    align-items: center;
    margin-top: -20px;
    margin-bottom: 12px;
    }

    .app-title-center {
        text-align: center !important;
        font-size: 2.6rem !important;
        font-weight: 900 !important;
        color: #2f3140 !important;
        margin: 0 !important;
        line-height: 1.1 !important;
    }

    .app-version-right {
        text-align: right;
        color: #7b8190;
        font-size: 0.9rem;
        font-weight: 500;
        padding-top: 10px;
    }

    .start-card {
        background: linear-gradient(90deg, #eaf3ff 0%, #f8fbff 100%);
        border: 1px solid #d6e8ff;
        border-radius: 16px;
        padding: 14px 18px;
        margin: 8px 0 14px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
    }

    .start-card-text {
        color: #0f2f7a;
        font-size: 1rem;
        line-height: 1.35;
    }

    .start-card-action {
        min-width: 210px;
        display: flex;
        justify-content: flex-end;
    }

    .start-card-action div[data-testid="stButton"] button {
        background: #2563eb !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.2rem !important;
        font-weight: 700 !important;
        border: none !important;
    }

    hr {
        margin-top: 10px !important;
        margin-bottom: 14px !important;
    }

    @media (max-width: 900px) {
        .top-header {
            grid-template-columns: 1fr;
            gap: 8px;
    }

    .app-version-right {
        text-align: center;
    }

    .start-card {
        flex-direction: column;
        align-items: stretch;
    }

    .start-card-action {
        justify-content: stretch;
        min-width: 100%;
    }
                
    /* BOTÃO PRIMÁRIO (Iniciar) */
    .btn-primary-wrap div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: #ffffff;
    border-radius: 12px;
    padding: 0.75rem 1.4rem;
    font-weight: 700;
    font-size: 0.95rem;
    border: none;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25);
    transition: all 0.25s ease;
    }

    .btn-primary-wrap div[data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.35);
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    }


    /* BOTÃO SECUNDÁRIO (Reiniciar) */
    .btn-secondary-wrap div[data-testid="stButton"] button {
        background: #ffffff;
        color: #1d4ed8;
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1.5px solid #c7d7f5;
        transition: all 0.2s ease;
    }

    .btn-secondary-wrap div[data-testid="stButton"] button:hover {
        background: #eef4ff;
        border-color: #1d4ed8;
        color: #1d4ed8;
    }


    /* ALINHAMENTO DENTRO DO CARD */
    .start-card-action {
        display: flex;
        gap: 10px;
        align-items: center;
    }
                
    .start-card-shell {
    background: linear-gradient(90deg, #eef6ff 0%, #f8fbff 100%);
    border: 1px solid #cfe3ff;
    border-radius: 18px;
    padding: 18px 22px;
    margin: 14px 0 18px 0;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
    }

    .start-card-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0f2f7a;
        margin-bottom: 4px;
    }

    .start-card-subtitle {
        font-size: 0.98rem;
        color: #1e3a8a;
    }

    .btn-primary-wrap button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border-radius: 14px !important;
        height: 48px !important;
        font-weight: 800 !important;
        border: none !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.28) !important;
    }

    .btn-primary-wrap button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    }

    .btn-secondary-wrap button {
        background: #ffffff !important;
        color: #1d4ed8 !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 14px !important;
        height: 48px !important;
        font-weight: 800 !important;
    }


}

    </style>
    """, unsafe_allow_html=True)

    components.html("""
        <script>
        document.documentElement.setAttribute("translate", "no");
        document.body.setAttribute("translate", "no");
        </script>
        """, height=0)

    load_environment()
    init_state()
    render_header()
    render_status()
    render_card_explicacao_chat()
    render_banner_topo_chat()
    render_chat_messages()
    render_confirmar_quiz_buttons()
    save_result_to_file()

    if st.session_state.started and st.session_state.phase not in ["extra", "unidade", "confirmar_quiz", "mostrar_resultado"]:        
        user_text = st.chat_input("Digite sua mensagem...")
    
        if user_text:
            with st.chat_message("assistant"):
                st.markdown("**IA For HEALTH:**")
                st.markdown(
                    """
                    <div style="display:flex;align-items:center;gap:10px;background:#ffffff;border:1px solid #dbeafe;border-radius:14px;padding:12px 14px;color:#334155;width:fit-content;box-shadow:0 4px 14px rgba(37,99,235,0.08);">
                        <span style="font-size:1.1rem;">⏳</span>
                        <span style="font-size:0.96rem;font-weight:500;">Pensando...</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            try:
                process_user_message(user_text)
            except Exception as exc:
                add_msg("assistant", f"Ocorreu um erro ao processar sua mensagem: {exc}", section="post_result")

            st.rerun()

bloquear_tradutor_google()
render_footer_fixo()   

run_scroll_to_result_script()
run_scroll_to_bottom_script()
run_focus_script()   


if __name__ == "__main__":
    main()
