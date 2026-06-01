import os
import re
import math
import random
from collections import Counter
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

APP_TITLE = "IA For HEALTH - Santa Casa BH"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(BASE_DIR, "resultados.txt")
KNOWLEDGE_FILE = "rag_santa_casa_bh_2024.txt"
KNOWLEDGE_DIR = "knowledge_base"
RAG_CHUNK_SIZE = 900
RAG_CHUNK_OVERLAP = 180
RAG_TOP_K = 4
APP_VERSION = "v8.3 - 01/06/2026 - RAG + OpenAI"

HEART_ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="36" height="36"><path d="M50 85 C50 85 10 58 10 32 C10 18 22 8 35 12 C41 14 46 18 50 24 C54 18 59 14 65 12 C78 8 90 18 90 32 C90 58 50 85 50 85Z" fill="#6300AB"/></svg>"""
USER_ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="32" height="32"><circle cx="20" cy="20" r="20" fill="#f3e8ff"/><circle cx="20" cy="15" r="7" fill="#6300AB"/><path d="M6 36 C6 27 12 22 20 22 C28 22 34 27 34 36" fill="#6300AB"/><rect x="16" y="18" width="8" height="3" rx="1" fill="#fff" opacity="0.7"/></svg>"""

# Frases de incentivo para múltipla escolha
FRASES_INCENTIVO_MC = [
    "🎯 Arrasou! Cada resposta conta nessa jornada!",
    "⭐ Muito bem! Você está indo super bem!",
    "🚀 Isso aí! Continue assim, você está mandando ver!",
    "💡 Show! Sua curiosidade sobre IA é incrível!",
    "🎉 Perfeito! Você está no caminho certo!",
]

FRASES_INCENTIVO_ABERTA = [
    "✍️ Muito bem! Sua visão sobre o tema ficou ótima!",
    "💬 Legal demais sua resposta! Você pensa muito bem!",
    "🌟 Adorei como você expressou isso! Continue assim!",
]

FRASES_QUASE_CERTO = [
    "😊 Quase lá! Sua resposta tocou no ponto certo, só faltou um detalhe — mas tá ótimo, qualquer hora você chega lá!",
    "🤗 Boa tentativa! Você chegou bem perto, com mais um tempinho você bate na mosca!",
    "😄 Legal! Você captou a ideia, a resposta só pedia um pouquinho mais de detalhe — sem pressão!",
]

def load_environment():
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    except Exception:
        st.error("OPENAI_API_KEY não encontrada nas Secrets do Streamlit.")
        st.stop()


def connect_google_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"], scope
    )
    client = gspread.authorize(creds)
    return client.open("Resultados_IA").worksheet("Resultados")


@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)


def read_text_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read().strip()


def normalize_text(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-zà-ú0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    stopwords = {
        "a","o","as","os","um","uma","uns","umas","de","da","do","das","dos",
        "e","em","no","na","nos","nas","para","por","com","sem","sobre","que",
        "qual","quais","quem","quando","onde","como","porque","porquê","é","ser",
        "são","foi","sua","seu","suas","seus","ao","à","aos","às","me","minha",
        "meu","tem","ter","pode","posso","vou","vai","santa","casa","bh"
    }
    return [w for w in normalize_text(text).split() if len(w) >= 3 and w not in stopwords]


def unique_list(items):
    seen, out = set(), []
    for i in items:
        if i not in seen:
            out.append(i); seen.add(i)
    return out


def remove_fonte_lines(answer):
    if not answer:
        return answer
    return "\n".join(l for l in answer.splitlines() if not l.strip().lower().startswith("fonte:")).strip()


def chunk_text(text, chunk_size=RAG_CHUNK_SIZE, overlap=RAG_CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    chunks, start = [], 0
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = []
    for p in [os.path.join(base_dir, KNOWLEDGE_FILE), os.path.join(os.getcwd(), KNOWLEDGE_FILE), KNOWLEDGE_FILE]:
        if os.path.isfile(p) and p not in paths:
            paths.append(p)
    for folder in [os.path.join(base_dir, KNOWLEDGE_DIR), os.path.join(os.getcwd(), KNOWLEDGE_DIR), KNOWLEDGE_DIR]:
        if os.path.isdir(folder):
            for fn in os.listdir(folder):
                if fn.lower().endswith(".txt"):
                    fp = os.path.join(folder, fn)
                    if fp not in paths:
                        paths.append(fp)
    return paths


@st.cache_data(show_spinner=False)
def build_rag_index():
    docs = []
    for path in get_knowledge_paths():
        try:
            text = read_text_file(path)
        except Exception:
            continue
        if not text:
            continue
        for idx, chunk in enumerate(chunk_text(text)):
            docs.append({"source": os.path.basename(path), "chunk_id": idx+1, "text": chunk, "tokens": tokenize(chunk)})
    return docs


def retrieve_context(question, top_k=RAG_TOP_K):
    docs = build_rag_index()
    if not docs:
        return "", []
    q_tokens = tokenize(question)
    if not q_tokens:
        return "", []
    total_docs = len(docs)
    df = Counter(t for doc in docs for t in set(doc["tokens"]))
    q_counter = Counter(q_tokens)
    scored = []
    for doc in docs:
        cc = Counter(doc["tokens"])
        score = sum(q_counter[t] * cc[t] * (math.log((1+total_docs)/(1+df[t]))+1) for t in q_counter if t in cc)
        if normalize_text(question) in normalize_text(doc["text"]):
            score += 5
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [d for _, d in scored[:top_k]]
    parts, sources, used = [], [], set()
    for doc in selected:
        label = f"RAG_SantaCasaBH - trecho {doc['chunk_id']}"
        if label in used:
            continue
        used.add(label); sources.append(label)
        parts.append(f"[Fonte interna: {label}]\n{doc['text']}")
    return "\n\n---\n\n".join(parts), unique_list(sources)


def is_internal_santa_casa_question(question):
    q = normalize_text(question)
    return any(t in q for t in ["santa casa","santa casa bh","hospital santa casa","francisco sales","santa efigenia","santa efigênia"])


def retrieve_context_blindado(question, top_k=RAG_TOP_K):
    docs = build_rag_index()
    if not docs:
        return "", []
    q_norm = normalize_text(question)
    q_tokens = set(tokenize(question))
    intent = get_question_intent(question)
    strong_terms = intent_terms(intent)
    scored = []
    for doc in docs:
        text_norm = normalize_text(doc.get("text",""))
        doc_tokens = set(doc.get("tokens",[]))
        score = len(q_tokens.intersection(doc_tokens)) * 2
        score += sum(8 for t in strong_terms if normalize_text(t) in text_norm)
        if is_internal_santa_casa_question(question):
            if any(t in text_norm for t in ["santa casa","hospital","santa efigenia","santa efigênia"]):
                score += 5
        if q_norm and q_norm in text_norm:
            score += 10
        if score > 0:
            scored.append((score, doc))
    if not scored:
        return retrieve_context(question, top_k=top_k)
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [d for _, d in scored[:top_k]]
    parts, sources, used = [], [], set()
    for doc in selected:
        label = f"RAG_SantaCasaBH - trecho {doc['chunk_id']}"
        if label in used:
            continue
        used.add(label); sources.append(label)
        parts.append(f"[Fonte interna: {label}]\n{doc['text']}")
    return "\n\n---\n\n".join(parts), unique_list(sources)


def get_question_intent(question):
    q = normalize_text(question)
    if any(t in q for t in ["endereco","endereço","onde fica","localizacao","localização","localizada","cep"]):
        return "endereco"
    if any(t in q for t in ["telefone","contato","ligar","numero","número"]):
        return "telefone"
    if any(t in q for t in ["primeiro nome","nome antigo","chamava","abrigo"]):
        return "primeiro_nome"
    if any(t in q for t in ["fundacao","fundação","fundada","fundado","existencia","existência","anos"]):
        return "fundacao"
    return "geral"


def intent_terms(intent):
    return {
        "endereco": ["endereço","endereco","localizada","avenida","rua","cep","bairro"],
        "telefone": ["telefone","contato","informações","informacoes"],
        "primeiro_nome": ["primeiro nome","hospital de abrigo","abrigo"],
        "fundacao": ["fundação","fundacao","fundada","fundado","existência","existencia","anos"],
        "geral": [],
    }.get(intent, [])


def read_all_knowledge_text():
    parts = []
    for path in get_knowledge_paths():
        try:
            t = read_text_file(path)
            if t:
                parts.append(t)
        except Exception:
            pass
    return "\n\n".join(parts).strip()


def direct_answer_from_raw_knowledge(question):
    raw = read_all_knowledge_text()
    if not raw:
        return "", ""
    intent = get_question_intent(question)
    raw_clean = re.sub(r"\s+", " ", re.sub(r"#+", " ", raw)).strip()
    if intent == "endereco" and is_internal_santa_casa_question(question):
        m = re.search(r"(A\s+Santa\s+Casa\s+BH\s+est[áa]\s+localizada\s+.*?(?:CEP\s*[\d\.\-]+))", raw_clean, re.IGNORECASE)
        if m:
            return m.group(1).strip(), "RAG_SantaCasaBH - trecho 1"
        m = re.search(r"(Avenida\s+[^.]+(?:CEP\s*[\d\.\-]+)?)", raw_clean, re.IGNORECASE)
        if m:
            return m.group(1).strip(), "RAG_SantaCasaBH - trecho 1"
    if intent == "telefone" and is_internal_santa_casa_question(question):
        m = re.search(r"(\(?\d{2}\)?\s*\d{4,5}[-\s]?\d{4})", raw_clean, re.IGNORECASE)
        if m:
            return f"O telefone geral para contato e informações é {m.group(1).strip()}.", "RAG_SantaCasaBH - trecho 1"
    return "", ""


def format_answer_with_source(answer, sources_text):
    answer = remove_fonte_lines(answer or "").strip()
    if not answer:
        answer = "Encontrei essa informação na base interna, mas não consegui montar uma resposta textual."
    return f"{answer}\n\nFonte: {sources_text}"


def answer_free_chat(user_text):
    llm = get_llm()
    pergunta = user_text.strip()
    pergunta_norm = normalize_text(pergunta)
    is_sc = is_internal_santa_casa_question(pergunta)
    if any(t in pergunta_norm for t in ["ela","dela","fundada","fundou","provedor","historia","história"]):
        is_sc = True
    if is_sc:
        rag_ctx, rag_src = retrieve_context_blindado(pergunta)
        df_raw, src_raw = direct_answer_from_raw_knowledge(pergunta)
        if df_raw:
            return format_answer_with_source(df_raw, src_raw)
        if rag_ctx and rag_src:
            sources_text = ", ".join(unique_list(rag_src))
            res = llm.invoke(f"""Você é a IA For HEALTH da Santa Casa BH.
Responda SOMENTE com base no CONTEXTO abaixo. Não invente. Seja curto e direto.
Se não estiver no contexto, responda: Essa informação não foi localizada na base interna da Santa Casa BH.

CONTEXTO:
{rag_ctx}

PERGUNTA:
{pergunta}

RESPOSTA FINAL:""")
            resp = res.content.strip() if res.content else ""
            if resp and "não foi localizada" not in resp.lower():
                return format_answer_with_source(resp, sources_text)
        return "Essa informação não foi localizada na base interna da Santa Casa BH.\n\nPara responder corretamente, atualize o arquivo rag_santa_casa_bh_2024.txt."
    res = llm.invoke(f"""Você é a IA For HEALTH, assistente profissional e objetivo.
Responda em português do Brasil. Seja claro, educado e direto.

Pergunta:
{pergunta}""")
    return res.content.strip() if res.content else "Não consegui responder agora."


class AnswerEvaluation(BaseModel):
    correct: bool
    score: int
    feedback: str


UNIDADES_NEGOCIO = [
    "Santa Casa BH","São Lucas","Centro de Autismo","Funerária e Assistência Familiar",
    "Faculdade de Saúde","Ambulatórios Especializados","Instituto Geriátrico",
    "Instituto Materno Infantil","Instituto de Oncologia","Pesquisa Clínica","Órix Lab","Corporativo"
]

QUESTION_BANK = [
    {"type":"multiple_choice","question":"O que é Inteligência Artificial?","options":["A) Um tipo de hardware de computador","B) Uma tecnologia que permite máquinas realizarem tarefas que simulam a inteligência humana","C) Um aplicativo de celular","D) Um sistema de armazenamento de dados"],"correct_option":1,"reference":"Uma tecnologia que permite máquinas realizarem tarefas que simulam a inteligência humana."},
    {"type":"multiple_choice","question":"Qual é um exemplo de uso de IA no dia a dia?","options":["A) Escrever em papel","B) Usar uma calculadora simples","C) Assistentes virtuais como Alexa e Siri","D) Ligar e desligar o computador"],"correct_option":2,"reference":"Assistentes virtuais como Alexa e Siri são exemplos de IA no dia a dia."},
    {"type":"multiple_choice","question":"O que a IA generativa faz?","options":["A) Apenas organiza dados","B) Cria conteúdos novos, como textos e imagens","C) Armazena arquivos","D) Executa cálculos simples"],"correct_option":1,"reference":"A IA generativa cria conteúdos novos, como textos e imagens."},
    {"type":"multiple_choice","question":"Qual é a principal função do machine learning?","options":["A) Criar internet","B) Ajudar máquinas a aprender com dados","C) Construir computadores","D) Instalar programas"],"correct_option":1,"reference":"A principal função do machine learning é ajudar máquinas a aprender com dados."},
    {"type":"multiple_choice","question":"Qual destes é um exemplo de IA no trabalho?","options":["A) Usar papel e caneta","B) Enviar carta pelo correio","C) Chatbots para atendimento ao cliente","D) Arquivar documentos em arquivos físicos"],"correct_option":2,"reference":"Chatbots para atendimento ao cliente são um exemplo de IA no trabalho."},
    {"type":"multiple_choice","question":"IA pode ajudar empresas a:","options":["A) Diminuir produtividade","B) Aumentar custos sempre","C) Automatizar tarefas repetitivas","D) Parar processos"],"correct_option":2,"reference":"A IA pode ajudar empresas a automatizar tarefas repetitivas."},
    {"type":"multiple_choice","question":"Qual é um risco do uso da IA?","options":["A) Melhorar eficiência","B) Ajudar na análise de dados","C) Uso inadequado de informações","D) Reduzir erros"],"correct_option":2,"reference":"Um risco do uso da IA é o uso inadequado de informações."},
    {"type":"multiple_choice","question":"O que é um chatbot?","options":["A) Um robô físico","B) Um sistema que conversa com pessoas por texto ou voz","C) Um tipo de computador","D) Um programa de edição de imagem"],"correct_option":1,"reference":"Um chatbot é um sistema que conversa com pessoas por texto ou voz."},
    {"type":"multiple_choice","question":"Qual é uma boa prática ao usar IA?","options":["A) Compartilhar qualquer dado sensível","B) Usar sem verificar informações","C) Revisar os resultados gerados","D) Confiar 100% em tudo que a IA gera"],"correct_option":2,"reference":"Uma boa prática ao usar IA é revisar os resultados gerados."},
    {"type":"multiple_choice","question":"IA tradicional geralmente:","options":["A) Aprende sozinha sem dados","B) Usa regras e padrões definidos","C) Cria imagens automaticamente sempre","D) Substitui completamente humanos"],"correct_option":1,"reference":"A IA tradicional geralmente usa regras e padrões definidos."},
    {"type":"multiple_choice","question":"IA generativa pode ser usada para:","options":["A) Apenas calcular números","B) Criar textos e imagens","C) Somente armazenar dados","D) Imprimir documentos"],"correct_option":1,"reference":"A IA generativa pode ser usada para criar textos e imagens."},
    {"type":"multiple_choice","question":"Qual dessas atividades pode ser automatizada com IA?","options":["A) Dormir","B) Analisar grandes volumes de dados","C) Comer","D) Respirar"],"correct_option":1,"reference":"A atividade de analisar grandes volumes de dados pode ser automatizada com IA."},
    {"type":"multiple_choice","question":"Por que empresas usam IA?","options":["A) Para complicar processos","B) Para reduzir eficiência","C) Para melhorar decisões e produtividade","D) Para evitar tecnologia"],"correct_option":2,"reference":"Empresas usam IA para melhorar decisões e produtividade."},
    {"type":"multiple_choice","question":"O que significa \"dados\" no contexto de IA?","options":["A) Apenas imagens","B) Informações usadas para treinar sistemas","C) Somente textos longos","D) Arquivos apagados"],"correct_option":1,"reference":"No contexto de IA, dados são informações usadas para treinar sistemas."},
    {"type":"multiple_choice","question":"Qual é um cuidado importante ao usar IA?","options":["A) Ignorar erros","B) Não revisar resultados","C) Proteger dados pessoais","D) Compartilhar tudo"],"correct_option":2,"reference":"Um cuidado importante ao usar IA é proteger dados pessoais."},
    {"type":"multiple_choice","question":"IA pode ajudar na área de atendimento ao cliente ao:","options":["A) Diminuir respostas","B) Ignorar clientes","C) Responder perguntas automaticamente","D) Fechar canais de atendimento"],"correct_option":2,"reference":"A IA pode ajudar no atendimento ao cliente respondendo perguntas automaticamente."},
    {"type":"multiple_choice","question":"Qual é uma limitação da IA?","options":["A) Nunca erra","B) Sempre entende contexto perfeitamente","C) Pode cometer erros e gerar informações incorretas","D) Funciona sem dados"],"correct_option":2,"reference":"Uma limitação da IA é que ela pode cometer erros e gerar informações incorretas."},
    {"type":"multiple_choice","question":"Machine learning depende principalmente de:","options":["A) Energia elétrica","B) Dados para aprender","C) Internet apenas","D) Alguns usuários"],"correct_option":1,"reference":"Machine learning depende principalmente de dados para aprender."},
    {"type":"multiple_choice","question":"IA responsável significa:","options":["A) Usar sem regras","B) Ignorar impactos","C) Usar de forma ética e segura","D) Usar apenas para diversão"],"correct_option":2,"reference":"IA responsável significa usar de forma ética e segura."},
    {"type":"multiple_choice","question":"Um benefício da IA nas empresas é:","options":["A) Aumentar erros","B) Reduzir produtividade","C) Ajudar na tomada de decisão","D) Eliminar todos os empregos"],"correct_option":2,"reference":"Um benefício da IA nas empresas é ajudar na tomada de decisão."},
    {"type":"multiple_choice","question":"O que significa 'Prompt' no contexto de IA?","options":["A) Um tipo de vírus","B) Uma instrução ou texto dado à IA para gerar uma resposta","C) Um programa de computador","D) Um tipo de memória"],"correct_option":1,"reference":"Prompt é uma instrução ou texto dado à IA para orientar a geração de uma resposta."},
    {"type":"multiple_choice","question":"O que é um modelo de linguagem grande (LLM)?","options":["A) Um banco de dados","B) Um sistema de IA treinado em grandes volumes de texto","C) Um programa de tradução","D) Um hardware específico"],"correct_option":1,"reference":"LLM é um sistema de IA treinado em grandes volumes de texto para compreender e gerar linguagem."},
    {"type":"multiple_choice","question":"Qual dessas é uma aplicação de IA na saúde?","options":["A) Controle de temperatura de salas","B) Análise de imagens médicas para diagnóstico","C) Reserva de leitos por telefone","D) Impressão de receitas"],"correct_option":1,"reference":"A análise de imagens médicas para auxílio no diagnóstico é uma aplicação de IA na saúde."},
    {"type":"multiple_choice","question":"O que é automação com IA?","options":["A) Substituir máquinas por humanos","B) Usar IA para executar tarefas repetitivas automaticamente","C) Desligar sistemas automaticamente","D) Criar relatórios em papel"],"correct_option":1,"reference":"Automação com IA é o uso de inteligência artificial para executar tarefas repetitivas de forma automática."},
    {"type":"multiple_choice","question":"Qual é a principal vantagem da IA na análise de dados?","options":["A) Reduz o volume de dados","B) Processa grandes volumes rapidamente e encontra padrões","C) Imprime relatórios mais bonitos","D) Armazena dados em nuvem"],"correct_option":1,"reference":"A IA processa grandes volumes de dados rapidamente e identifica padrões que seriam difíceis para humanos."},
    {"type":"multiple_choice","question":"O que é 'bias' (viés) em IA?","options":["A) Um erro de digitação","B) Preconceito embutido nos dados que pode gerar resultados injustos","C) Um tipo de algoritmo","D) A velocidade de processamento"],"correct_option":1,"reference":"Viés em IA é quando os dados de treinamento contêm preconceitos que se refletem nos resultados do modelo."},
    {"type":"multiple_choice","question":"O que faz um assistente virtual com IA?","options":["A) Armazena arquivos","B) Responde perguntas e executa tarefas por comandos de voz ou texto","C) Conecta à internet","D) Imprime documentos"],"correct_option":1,"reference":"Um assistente virtual com IA responde perguntas e executa tarefas por comandos de voz ou texto."},
    {"type":"multiple_choice","question":"Para que serve a IA generativa no dia a dia profissional?","options":["A) Para apagar e-mails antigos","B) Para criar textos, resumos, imagens e conteúdos de forma rápida","C) Para instalar programas","D) Para bloquear sites"],"correct_option":1,"reference":"A IA generativa cria textos, resumos, imagens e conteúdos, agilizando tarefas do dia a dia profissional."},
    {"type":"multiple_choice","question":"O que é RAG (Retrieval-Augmented Generation)?","options":["A) Um tipo de vírus de computador","B) Uma técnica que combina busca de informações com geração de texto pela IA","C) Um formato de arquivo","D) Um protocolo de rede"],"correct_option":1,"reference":"RAG é uma técnica que combina busca de informações externas com geração de texto pela IA para respostas mais precisas."},
    {"type":"multiple_choice","question":"Qual atitude é correta ao receber uma informação gerada por IA?","options":["A) Publicar imediatamente sem verificar","B) Ignorar a informação","C) Verificar a informação em fontes confiáveis antes de usar","D) Confiar cegamente na IA"],"correct_option":2,"reference":"A atitude correta é verificar a informação em fontes confiáveis antes de utilizá-la."},
    {"type":"multiple_choice","question":"O que é deep learning?","options":["A) Aprendizado com livros físicos","B) Um tipo de machine learning que usa redes neurais profundas","C) Um idioma de programação","D) Um banco de dados especial"],"correct_option":1,"reference":"Deep learning é um tipo de machine learning que usa redes neurais com muitas camadas para aprender padrões complexos."},
    {"type":"multiple_choice","question":"Como a IA pode ajudar no atendimento ao paciente em hospitais?","options":["A) Substituindo completamente os médicos","B) Realizando cirurgias sozinha","C) Triando sintomas e direcionando pacientes mais rapidamente","D) Emitindo laudos sem revisão humana"],"correct_option":2,"reference":"A IA pode triagem de sintomas e direcionar pacientes, agilizando o atendimento sem substituir profissionais."},
    {"type":"multiple_choice","question":"O que significa IA explicável (Explainable AI)?","options":["A) IA que fala em voz alta","B) IA cujas decisões podem ser compreendidas e explicadas por humanos","C) IA que funciona sem internet","D) IA que aprende mais rápido"],"correct_option":1,"reference":"IA explicável é aquela cujas decisões podem ser compreendidas e justificadas por humanos."},
    {"type":"multiple_choice","question":"O que é processamento de linguagem natural (NLP)?","options":["A) Um método de compressão de arquivos","B) Capacidade da IA de entender e gerar linguagem humana","C) Um tipo de banco de dados","D) Um protocolo de segurança"],"correct_option":1,"reference":"NLP é a área da IA que permite às máquinas entender, interpretar e gerar linguagem humana."},
    {"type":"multiple_choice","question":"Qual é um exemplo de uso inadequado da IA?","options":["A) Resumir documentos longos","B) Traduzir textos","C) Criar notícias falsas para enganar pessoas","D) Organizar agendas"],"correct_option":2,"reference":"Criar notícias falsas é um exemplo de uso inadequado e antiético da IA."},
    {"type":"multiple_choice","question":"O que é um agente de IA?","options":["A) Um funcionário que usa IA","B) Um sistema de IA capaz de tomar ações autonomamente para atingir objetivos","C) Um aplicativo de mensagens","D) Um tipo de servidor"],"correct_option":1,"reference":"Um agente de IA é um sistema capaz de tomar ações autonomamente para atingir objetivos definidos."},
    {"type":"multiple_choice","question":"Por que é importante ter diversidade nos dados de treino da IA?","options":["A) Para deixar o sistema mais lento","B) Para garantir que a IA seja justa e represente diferentes grupos","C) Para usar mais memória","D) Para dificultar o acesso"],"correct_option":1,"reference":"Diversidade nos dados de treino garante que a IA seja mais justa e representativa de diferentes grupos."},
    {"type":"multiple_choice","question":"O que é fine-tuning em IA?","options":["A) Atualizar o hardware do computador","B) Ajustar um modelo pré-treinado para uma tarefa ou domínio específico","C) Limpar o banco de dados","D) Instalar atualizações do sistema"],"correct_option":1,"reference":"Fine-tuning é o processo de ajustar um modelo pré-treinado para uma tarefa ou domínio específico."},
    {"type":"multiple_choice","question":"O que são embeddings em IA?","options":["A) Tipos de vírus","B) Representações numéricas de palavras ou dados que capturam seu significado","C) Formatos de vídeo","D) Protocolos de rede"],"correct_option":1,"reference":"Embeddings são representações numéricas de palavras ou dados que capturam relações de significado."},
    {"type":"multiple_choice","question":"Qual é o papel humano mais importante quando se usa IA em decisões críticas?","options":["A) Apenas apertar botões","B) Supervisionar, questionar e validar as saídas da IA","C) Ignorar as recomendações da IA","D) Deixar a IA decidir sozinha sempre"],"correct_option":1,"reference":"O papel humano mais importante é supervisionar, questionar e validar as saídas da IA em decisões críticas."},
    {"type":"open","question":"Como a IA pode ajudar a reduzir o tempo gasto em tarefas administrativas no trabalho?","options":[],"correct_option":None,"reference":"Automatizando tarefas simples e repetitivas, organizando informações de forma mais rápida e economizando tempo em processos rotineiros.","keywords":["automatizar tarefas","tarefas repetitivas","economizar tempo","organizar informações"]},
    {"type":"open","question":"O que pode acontecer se você usar uma resposta de IA sem revisar antes?","options":[],"correct_option":None,"reference":"Pode conter erros ou informações incorretas, gerando problemas por falta de verificação e revisão.","keywords":["erros","informação incorreta","falta de revisão","risco"]},
    {"type":"open","question":"Como a IA pode ajudar na organização do seu trabalho diário?","options":[],"correct_option":None,"reference":"Sugerindo formas de organizar tarefas e prioridades, ajudando a estruturar atividades e ideias, e auxiliando no planejamento.","keywords":["organizar tarefas","prioridades","estruturação","planejamento"]},
    {"type":"open","question":"Qual cuidado você deve ter ao usar IA para gerar textos profissionais?","options":[],"correct_option":None,"reference":"Garantir que o conteúdo esteja adequado ao contexto, revisar e ajustar o texto antes de usar.","keywords":["revisar conteúdo","adequação","contexto","ajustes"]},
    {"type":"open","question":"Como a IA pode apoiar a criatividade no trabalho?","options":[],"correct_option":None,"reference":"Sugerindo ideias iniciais para projetos, ajudando a criar conteúdos mais rápidos e oferecendo apoio criativo.","keywords":["sugestões","ideias","criatividade","apoio"]},
    {"type":"open","question":"Quais são os riscos de compartilhar informações sensíveis com IA?","options":[],"correct_option":None,"reference":"Os dados podem ser expostos ou usados de forma inadequada, comprometendo a segurança das informações e podendo causar vazamentos.","keywords":["dados sensíveis","segurança","vazamento","exposição"]},
    {"type":"open","question":"Como a IA pode ajudar na comunicação dentro de uma empresa?","options":[],"correct_option":None,"reference":"Ajudando a resumir mensagens ou documentos, facilitando a criação de textos claros e melhorando a comunicação interna.","keywords":["resumo","comunicação","clareza","mensagens"]},
    {"type":"open","question":"Por que a IA não deve ser usada sozinha em decisões importantes?","options":[],"correct_option":None,"reference":"Porque pode cometer erros e não considerar todos os fatores, sendo necessária a avaliação humana em decisões importantes.","keywords":["decisão humana","erro","limitação","avaliação"]},
    {"type":"open","question":"Como a IA pode ajudar você a aprender algo novo no trabalho?","options":[],"correct_option":None,"reference":"Explicando conceitos de forma simples, resumindo conteúdos difíceis e facilitando o aprendizado de novos conhecimentos.","keywords":["aprendizado","explicação","resumo","conhecimento"]},
    {"type":"open","question":"Quais são sinais de que uma resposta de IA pode não ser confiável?","options":[],"correct_option":None,"reference":"Quando a informação parece incoerente ou confusa, ou quando não há confirmação em outras fontes confiáveis.","keywords":["incoerente","confuso","sem fonte","verificar"]},
]

TEMAS_COMPARTILHAMENTO = [
    "Fundamentos de IA","Machine Learning","Deep Learning","Processamento de Linguagem Natural (NLP)",
    "IA Generativa","Large Language Models (LLMs)","Engenharia de Prompt",
    "RAG (Retrieval-Augmented Generation)","Fine-Tuning de Modelos","Embeddings e Busca Vetorial",
    "Agentes de IA (AI Agents)","Speech AI e IA de Voz","IA Multimodal","Automação Inteligente",
    "IA Aplicada à Saúde","IA Aplicada a Negócios","IA para Produtividade","IA para Experiência do Usuário",
    "Governança de IA","Ética em IA","Segurança em IA","Compliance e LGPD em IA","Infraestrutura para IA",
    "APIs e Integrações de IA","Cloud AI","Sistemas Autônomos","Analytics Preditivo","IA Conversacional",
    "Sistemas Especialistas","Transformação Digital com IA","IA e Futuro do Trabalho","IA e Educação",
    "IA e Pesquisa Científica","IA e Cibersegurança","IA e IoT"
]


def init_state():
    defaults = {
        "started": False, "phase": "idle", "name": "", "matricula": "",
        "unidade_negocio": "", "aceitou_responder_quiz": "", "questions": [],
        "index": 0, "results": [], "score": 0, "chat": [],
        "extra_question_needed": False, "extra_question_answered": False,
        "extra_question_answer": "", "extra_question_topic": "",
        "focus_input": False, "scroll_to_result": False, "scroll_to_bottom": False,
        "quiz_completed": False, "test_start_time": None, "test_end_time": None,
        "result_saved": False, "extra_question_processing": False,
        "pending_scroll_bottom": False, "scroll_nonce": 0,
        "force_result_scroll_after_extra": False,
        # papel: "Instrutor", "Aluno", "Não"
        "papel_jornada": "",
        "fase_papel": False,  # True quando score >= 4 e quiz concluido
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_msg(role, text, section="quiz"):
    st.session_state.chat.append({"role": role, "content": text, "section": section})


def format_datetime(value):
    return value.strftime("%d/%m/%Y %H:%M:%S") if value else "-"


def save_result_to_file():
    if not st.session_state.quiz_completed or st.session_state.result_saved:
        return
    # Aguarda definição do papel quando score >= 4
    if st.session_state.get("fase_papel") and not st.session_state.extra_question_answered:
        return
    if st.session_state.test_end_time is None:
        st.session_state.test_end_time = datetime.now()
    try:
        sheet = connect_google_sheet()
        sheet.append_row([
            format_datetime(st.session_state.test_start_time),
            st.session_state.name,
            st.session_state.matricula,
            st.session_state.unidade_negocio,
            f"{st.session_state.score}/5",
            # Compartilhar Conh? → papel_jornada
            st.session_state.papel_jornada or "Não",
            st.session_state.extra_question_topic,
            format_datetime(st.session_state.test_end_time)
        ])
        st.toast("✅ Resultado salvo com sucesso!")
    except Exception as e:
        st.error(f"Erro ao salvar no Google Sheets: {str(e)}")
    st.session_state.result_saved = True


def reset_test_data():
    st.session_state.update({
        "phase": "name", "chat": [], "results": [], "index": 0, "score": 0,
        "name": "", "matricula": "", "unidade_negocio": "",
        "questions": [], "extra_question_needed": False, "extra_question_answered": False,
        "extra_question_answer": "", "extra_question_topic": "", "focus_input": True,
        "scroll_to_result": False, "scroll_to_bottom": True, "quiz_completed": False,
        "test_start_time": datetime.now(), "test_end_time": None, "result_saved": False,
        "extra_question_processing": False, "pending_scroll_bottom": False,
        "force_result_scroll_after_extra": False,
        "papel_jornada": "", "fase_papel": False,
        "scroll_nonce": st.session_state.get("scroll_nonce", 0) + 1,
    })


def start_conversation():
    st.session_state.started = True
    reset_test_data()
    add_msg("assistant", "Olá! 👋 Que ótimo ter você aqui! Vamos começar? Me diz seu nome completo (nome e sobrenome) 😊", section="quiz")
    st.session_state.focus_input = False
    st.session_state.scroll_to_bottom = True
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1


def reiniciar_teste():
    reset_test_data()
    add_msg("assistant", "Olá! 👋 Que ótimo ter você aqui! Vamos começar? Me diz seu nome completo (nome e sobrenome) 😊", section="quiz")
    st.rerun()


def responder_papel_jornada(papel):
    """Chamada quando colaborador escolhe Instrutor / Aluno / Não"""
    st.session_state.papel_jornada = papel
    add_msg("user", papel, section="quiz")
    st.session_state.scroll_to_bottom = True
    st.session_state.scroll_nonce += 1

    if papel in ("Instrutor", "Aluno"):
        st.session_state.extra_question_answered = False
        st.session_state.phase = "tema_compartilhamento"
        label = "ensinar" if papel == "Instrutor" else "aprender"
        add_msg("assistant",
            f"Incrível! 🎓 Que tal nos contar quais temas você quer {label}? "
            f"Selecione um ou mais assuntos abaixo! 👇",
            section="quiz")
    else:
        # Não deseja papel
        st.session_state.extra_question_answered = True
        st.session_state.phase = "chat"
        add_msg("assistant",
            f"Tudo bem! 😄 Sua participação foi registrada. "
            f"Lembra: o IA For Health é seu amigo e está sempre aqui pra tirar suas dúvidas sobre IA! 💜 "
            f"Pode perguntar à vontade no chat abaixo 👇",
            section="quiz")
        st.session_state.focus_input = True
        st.session_state.test_end_time = datetime.now()
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
    st.session_state.scroll_nonce += 1
    st.session_state.test_end_time = datetime.now()
    add_msg("assistant",
        "Temas registrados! 🎉 Isso é muito legal, sua contribuição vai fazer diferença na jornada de todo mundo! 💪 "
        "Agora o chat está livre — pode perguntar qualquer coisa sobre IA pra mim, tô aqui! 😊",
        section="quiz")
    save_result_to_file()
    st.rerun()


# ── SCRIPTS JS ────────────────────────────────────────────────────────────────
def render_footer_fixo():
    components.html("""<script>
    const doc=window.parent.document;
    function criarRodape(){
        let f=doc.getElementById("footer-santa-ideia-fixo");
        if(!f){f=doc.createElement("div");f.id="footer-santa-ideia-fixo";f.innerHTML="® Santa ideia 2026 - Órix Lab";doc.body.appendChild(f);}
        Object.assign(f.style,{position:"fixed",left:"0",right:"0",bottom:"4px",width:"100%",textAlign:"center",color:"#64748b",fontSize:"0.85rem",fontWeight:"700",zIndex:"2147483647",pointerEvents:"none",background:"transparent"});
    }
    criarRodape();setInterval(criarRodape,500);
    </script>""", height=0)


def run_focus_script():
    if st.session_state.get("scroll_to_bottom") or not st.session_state.get("focus_input"):
        return
    phase = st.session_state.get("phase","")
    selectors = '["textarea[data-testid=\'stChatInputTextArea\']","div[data-testid=\'stChatInput\'] textarea"]' if phase != "tema_compartilhamento" else '["input[type=\'text\']"]'
    components.html(f"""<script>
    const focusInput=()=>{{const doc=window.parent.document;const sels={selectors};for(const s of sels){{const el=doc.querySelector(s);if(el){{el.focus({{preventScroll:true}});return true;}}}}return false;}};
    let tries=0;const t=setInterval(()=>{{tries++;if(focusInput()||tries>40)clearInterval(t);}},150);
    </script>""", height=0)
    st.session_state.focus_input = False


def run_scroll_to_result_script():
    if not st.session_state.get("scroll_to_result"):
        return
    components.html("""<script>
(function() {
    function s() {
        var d = window.parent.document;
        var el = d.getElementById("card-resultado-avaliacao");
        if (el) { el.scrollIntoView({ behavior: "smooth", block: "start" }); }
    }
    [150, 500, 900].forEach(function(ms) { setTimeout(s, ms); });
})();
</script>""", height=0)
    st.session_state.scroll_to_result = False


def _render_scroll_anchor():
    should_scroll = st.session_state.get("scroll_to_bottom", False)
    nonce = st.session_state.get("scroll_nonce", 0)
    ts = datetime.now().timestamp()
    if should_scroll:
        st.session_state.scroll_to_bottom = False
    go = "true" if should_scroll else "false"
    components.html(f"""
<div id="anchor-{nonce}-{ts}" style="height:1px;width:1px;opacity:0;pointer-events:none;"></div>
<script>
(function() {{
    var GO = {go};
    function scrollToBottom() {{
        try {{
            var p = window.parent;
            var d = p.document;
            var fim = d.getElementById("fim-da-pagina");
            if (fim) {{ fim.scrollIntoView({{ behavior: "smooth", block: "end" }}); }}
            p.scrollTo({{ top: 999999, behavior: "smooth" }});
            d.documentElement.scrollTop = 999999;
            d.body.scrollTop = 999999;
            var selectors = [
                '[data-testid="stAppViewContainer"]',
                '[data-testid="stMain"]',
                '[data-testid="stMainBlockContainer"]',
                '.main', '.block-container'
            ];
            for (var i = 0; i < selectors.length; i++) {{
                var el = d.querySelector(selectors[i]);
                if (el) el.scrollTop = 999999;
            }}
        }} catch(e) {{
            try {{ window.scrollTo({{ top: 999999, behavior: "smooth" }}); }} catch(_) {{}}
        }}
    }}
    if (!GO) return;
    [50, 200, 500, 900, 1500].forEach(function(ms) {{ setTimeout(scrollToBottom, ms); }});
    try {{
        var target = window.parent.document.body;
        var lastFire = 0;
        var obs = new MutationObserver(function() {{
            var now = Date.now();
            if (now - lastFire < 300) return;
            lastFire = now;
            scrollToBottom();
        }});
        obs.observe(target, {{ childList: true, subtree: true }});
        setTimeout(function() {{ obs.disconnect(); }}, 4000);
    }} catch(e) {{}}
}})();
</script>
""", height=0)


# ── AVALIAÇÃO ─────────────────────────────────────────────────────────────────
def evaluate(question, reference, user_answer, keywords=None):
    if keywords:
        answer_norm = normalize_text(user_answer)
        hits = sum(1 for kw in keywords if normalize_text(kw) in answer_norm)
        if hits >= 1:
            feedback = random.choice(FRASES_INCENTIVO_ABERTA)
            return AnswerEvaluation(correct=True, score=1, feedback=feedback)
    llm = get_llm()
    keywords_str = ", ".join(keywords) if keywords else ""
    res = llm.invoke(f"""Você é um avaliador bem-humorado e acolhedor de conhecimento em IA para colaboradores da Santa Casa BH.
Avalie a resposta do usuário com base na resposta esperada e nas palavras-chave aceitas.
Considere CORRETA se a resposta mencionar pelo menos uma das palavras-chave ou abordar o conceito central, mesmo com palavras diferentes.
Se estiver parcialmente certa, considere correta também — o tom deve ser sempre leve, encorajador e sem pressão.
Se incorreta, diga de forma bem gentil e divertida que faltou um detalhe, sem nunca fazer o usuário se sentir mal.
Retorne EXATAMENTE neste formato:
correct: true ou false
score: 1 ou 0
feedback: texto curto em português com emoji, tom leve e encorajador

Pergunta: {question}
Resposta esperada: {reference}
Palavras-chave aceitas: {keywords_str}
Resposta do usuário: {user_answer}""")
    content = res.content.lower() if res.content else ""
    correct = "correct: true" in content
    score = 1 if correct else 0
    raw_feedback = res.content if res.content else ""
    # Extrai apenas o texto após "feedback:" (case-insensitive, pega a última ocorrência)
    if "feedback:" in raw_feedback.lower():
        idx = raw_feedback.lower().rfind("feedback:")
        raw_feedback = raw_feedback[idx + len("feedback:"):].strip()
    # Remove linhas residuais de "correct:" e "score:" que possam ter sobrado
    linhas = [l for l in raw_feedback.splitlines() if not l.strip().lower().startswith(("correct:", "score:"))]
    raw_feedback = " ".join(l.strip() for l in linhas if l.strip()).strip()
    # Se modelo não retornou feedback amigável, usa padrão
    if not raw_feedback or len(raw_feedback) < 5:
        raw_feedback = random.choice(FRASES_QUASE_CERTO) if not correct else random.choice(FRASES_INCENTIVO_ABERTA)
    return AnswerEvaluation(correct=correct, score=score, feedback=raw_feedback)


def contem_palavra_ofensiva(texto):
    return any(p in normalize_text(texto).split() for p in ["puta","fdp","burro","cu"])


def nome_valido(nome):
    nome = nome.strip()
    partes = [p for p in nome.split() if len(p) >= 2]
    return (
        len(nome) >= 5
        and len(partes) >= 2
        and not contem_palavra_ofensiva(nome)
        and bool(re.match(r"^[A-Za-zÀ-ÿ\s]+$", nome))
    )


def matricula_valida(matricula):
    matricula = matricula.strip()
    return matricula.isdigit() and 3 <= len(matricula) <= 6


def validar_nome_com_openai(nome):
    llm = get_llm()
    res = llm.invoke(f"""Você é um validador de cadastro. Analise se o texto é um nome próprio completo real (nome e sobrenome).
Texto: {nome}
Regras: Bloqueie palavrões, nomes estranhos, nomes incompletos (somente primeiro nome). Aceite nomes reais com acentos.
Responda somente:
valido: sim ou nao
motivo: texto curto""")
    resp = res.content.strip().lower() if res.content else ""
    valido = "valido: sim" in resp or "válido: sim" in resp
    motivo = resp.split("motivo:",1)[1].strip() if "motivo:" in resp else "Nome inválido."
    return valido, motivo


def process_name(user_text):
    # Normaliza capitalização: aceita minúscula, maiúscula ou mista
    nome_normalizado = user_text.strip().title()
    add_msg("user", user_text, section="quiz")
    if not nome_valido(nome_normalizado):
        add_msg("assistant",
            "Ops! 😅 Preciso do seu **nome completo** (nome e sobrenome) pra continuar. "
            "Pode digitar de novo? Prometo que é rapidinho! 😊",
            section="quiz")
        st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1; return
    valido_openai, motivo = validar_nome_com_openai(nome_normalizado)
    if not valido_openai:
        add_msg("assistant",
            "Hmm, não consegui identificar esse nome. 🤔 Tente colocar seu **nome e sobrenome** completos, tá? 😊",
            section="quiz")
        st.session_state.focus_input = True; st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1; return
    st.session_state.name = nome_normalizado
    st.session_state.phase = "matricula"
    st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1
    add_msg("assistant",
        f"Que nome bonito, {st.session_state.name}! 😄✨ Agora me diz sua matrícula, por favor:",
        section="quiz")


def process_matricula(user_text):
    add_msg("user", user_text, section="quiz")
    if not matricula_valida(user_text):
        add_msg("assistant",
            "Eita! 😄 Matrícula inválida. Coloca só os números, sem letras nem espaços. Pode tentar de novo? 👍",
            section="quiz")
        st.session_state.focus_input = True; st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1; return
    st.session_state.matricula = user_text.strip()
    st.session_state.phase = "unidade"
    st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1
    add_msg("assistant",
        "Ótimo! 🏥 Agora seleciona sua unidade de negócio abaixo 👇",
        section="quiz")


def process_unidade(unidade):
    if unidade not in UNIDADES_NEGOCIO:
        add_msg("assistant","Unidade inválida. Selecione uma das opções disponíveis.",section="quiz"); return
    st.session_state.unidade_negocio = unidade
    add_msg("user", unidade, section="quiz")
    st.session_state.phase = "confirmar_quiz"
    st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1
    add_msg("assistant",
        f"Show, {st.session_state.name}! 🎉 Tudo certo por aqui!\n\n"
        f"Agora vem a parte divertida! 🕹️ Preparei uma brincadeira com 5 fases sobre Inteligência Artificial — "
        f"é bem tranquilo, sem pressão nenhuma, só pra a gente explorar juntos o que você já sabe! 😊\n\n"
        f"Topa embarcar nessa aventura? 🚀",
        section="quiz")


def process_multiple_choice(option_index):
    current = st.session_state.questions[st.session_state.index]
    chosen_text = current["options"][option_index]
    correct = (option_index == current["correct_option"])
    if correct:
        feedback_text = random.choice(FRASES_INCENTIVO_MC)
    else:
        feedback_text = f"😅 Quase lá! A resposta era: {current['options'][current['correct_option']]} — mas tá indo bem!"
    add_msg("user", chosen_text, section="quiz")
    st.session_state.results.append({
        "question": current["question"], "reference": current["reference"],
        "user_answer": chosen_text, "correct": correct,
        "score": 1 if correct else 0, "feedback": feedback_text, "type": "multiple_choice",
    })
    st.session_state.index += 1
    st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1
    if st.session_state.index < len(st.session_state.questions):
        next_q = st.session_state.questions[st.session_state.index]
        st.session_state.phase = "quiz"
        fase_atual = st.session_state.index + 1
        total = len(st.session_state.questions)
        if next_q["type"] == "open":
            st.session_state.focus_input = True
            frases_abertas = [
                "✍️ Última fase! Agora é livre — escreve do seu jeito, sem neura! 😊",
                "🖊️ Chegou a fase final! Aqui você responde com suas próprias palavras, relaxa! 💜",
                "🌟 Última etapa! Sem resposta certa ou errada, só a sua visão conta! 🚀",
            ]
            add_msg("assistant", random.choice(frases_abertas), section="quiz")
        else:
            st.session_state.focus_input = False
            frases_transicao = [
                "💪 Isso aí! Vamos para a próxima! Você tá arrasando!",
                "🌟 Boa! Continua assim, tô na torcida por você!",
                "🚀 Partiu próxima fase! Você está voando!",
                "🎯 Mandou bem! Mais uma e já já chega lá!",
                "😄 Seguindo em frente! Cada fase é uma conquista!",
            ]
            add_msg("assistant", random.choice(frases_transicao), section="quiz")
        return
    _finalizar_quiz()


def process_quiz_answer(user_text):
    add_msg("user", user_text, section="quiz")
    current = st.session_state.questions[st.session_state.index]
    keywords = current.get("keywords", [])
    result = evaluate(current["question"], current["reference"], user_text.strip(), keywords=keywords)
    st.session_state.results.append({
        "question": current["question"], "reference": current["reference"],
        "user_answer": user_text.strip(), "correct": result.correct,
        "score": result.score, "feedback": result.feedback, "type": "open",
    })
    st.session_state.index += 1
    st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1
    _finalizar_quiz()


def _finalizar_quiz():
    total = sum(item["score"] for item in st.session_state.results)
    st.session_state.score = total
    st.session_state.quiz_completed = True
    st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1

    if total >= 4:
        # Score alto → pergunta sobre papel na jornada
        st.session_state.fase_papel = True
        st.session_state.extra_question_needed = True
        st.session_state.extra_question_answered = False
        st.session_state.phase = "mostrar_resultado"
        add_msg("assistant",
            f"Uau, {st.session_state.name}! 🎊 Você foi incrível nessa jornada! "
            f"Confira abaixo o resumo da sua aventura! 👇",
            section="quiz")
    else:
        st.session_state.fase_papel = False
        st.session_state.extra_question_needed = False
        st.session_state.extra_question_answered = True
        st.session_state.phase = "chat"
        st.session_state.focus_input = True
        st.session_state.test_end_time = datetime.now()
        add_msg("assistant",
            f"Parabéns, {st.session_state.name}! 🥳 Você completou todas as fases! "
            f"Confira abaixo o resumo da sua jornada! "
            f"Lembra: o IA For Health é seu amigo e tá sempre aqui pra tirar suas dúvidas! 💜",
            section="quiz")
        save_result_to_file()


def process_free_chat(user_text):
    add_msg("user", user_text, section="post_result")
    answer = answer_free_chat(user_text)
    add_msg("assistant", answer, section="post_result")
    st.session_state.focus_input = True
    st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1


def process_user_message(user_text):
    if not user_text or not user_text.strip():
        return
    phase = st.session_state.phase
    if phase == "name":
        process_name(user_text); return
    if phase == "matricula":
        process_matricula(user_text); return
    if phase == "unidade":
        process_unidade(user_text); return
    if phase == "quiz":
        idx = st.session_state.index
        if idx < len(st.session_state.questions) and st.session_state.questions[idx]["type"] == "open":
            process_quiz_answer(user_text)
        else:
            add_msg("assistant","Por favor, selecione uma das opções acima 👆",section="quiz")
        return
    if phase == "tema_compartilhamento":
        add_msg("assistant","Selecione um ou mais temas nas opções disponíveis e clique em Salvar temas 😊",section="quiz")
        st.session_state.scroll_to_bottom = True; return
    if phase == "chat":
        process_free_chat(user_text); return


# ── UI ────────────────────────────────────────────────────────────────────────
def render_header():
    heart_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 90" width="48" height="48" style="vertical-align:middle;margin-right:10px;"><path d="M50 85 C50 85 5 55 5 28 C5 14 17 5 30 9 C37 11 44 17 50 25 C56 17 63 11 70 9 C83 5 95 14 95 28 C95 55 50 85 50 85Z" fill="#6300AB"/></svg>'''
    st.markdown(f"""
    <div class="top-header">
    <div></div>
    <div><h1 class="app-title-center">{heart_svg}{APP_TITLE}</h1></div>
    <div class="app-version-right">Versão da IA: {APP_VERSION}</div>
    </div>""", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="start-card-shell">', unsafe_allow_html=True)
        col_texto, col_botao = st.columns([5, 1.2])
        with col_texto:
            st.markdown("""<div class="start-card-text">
            <div class="start-card-title">🚀 Pronto para a aventura?</div>
            <div class="start-card-subtitle">Embarque na jornada de IA e descubra o quanto você já sabe! 🧠💜</div>
            </div>""", unsafe_allow_html=True)
        with col_botao:
            if not st.session_state.started:
                st.markdown('<div class="btn-primary-wrap">', unsafe_allow_html=True)
                if st.button("🎮 Começar jornada", key="btn_iniciar_conversa", use_container_width=True):
                    start_conversation(); st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="btn-secondary-wrap">', unsafe_allow_html=True)
                if st.button("🔄 Reiniciar", key="btn_reiniciar_conversa", use_container_width=True):
                    nonce = st.session_state.get("scroll_nonce", 0) + 1
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.session_state.scroll_to_bottom = True
                    st.session_state.scroll_nonce = nonce
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.divider()


def render_status():
    if not st.session_state.started:
        return
    nome      = st.session_state.name or "-"
    matricula = st.session_state.matricula or "-"
    unidade   = st.session_state.unidade_negocio or "-"
    pontuacao = f"{st.session_state.score}/5" if st.session_state.quiz_completed else "-"
    st.markdown(f"""
    <div class="status-bar">
      <div class="status-item"><span class="status-label">Nome</span><span class="status-value">{nome}</span></div>
      <div class="status-item"><span class="status-label">Matrícula</span><span class="status-value">{matricula}</span></div>
      <div class="status-item"><span class="status-label">Unidade</span><span class="status-value">{unidade}</span></div>
      <div class="status-item"><span class="status-label">Pontuação</span><span class="status-value">{pontuacao}</span></div>
    </div>
    <hr style="margin:12px 0 8px 0;">
    """, unsafe_allow_html=True)


def render_chat_block(messages):
    ai_avatar = (
        '<div class="chat-avatar chat-avatar-ai">'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 90" width="28" height="28">'
        '<path d="M50 85 C50 85 5 55 5 28 C5 14 17 5 30 9 C37 11 44 17 50 25 '
        'C56 17 63 11 70 9 C83 5 95 14 95 28 C95 55 50 85 50 85Z" fill="#6300AB"/>'
        '</svg></div>'
    )
    user_avatar = (
        '<div class="chat-avatar chat-avatar-user">'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="28" height="28">'
        '<circle cx="20" cy="20" r="20" fill="#ede9fe"/>'
        '<circle cx="20" cy="14" r="7" fill="#6300AB"/>'
        '<path d="M4 38 Q4 26 20 26 Q36 26 36 38Z" fill="#6300AB"/>'
        '<rect x="14" y="18" width="12" height="3.5" rx="1.5" fill="white" opacity="0.8"/>'
        '</svg></div>'
    )
    html_parts = []
    for m in messages:
        role = m["role"]
        content = str(m.get("content", "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        if role == "assistant":
            html_parts.append(
                f'<div class="chat-row chat-row-ai">'
                f'{ai_avatar}'
                f'<div class="chat-bubble chat-bubble-ai">'
                f'<span class="chat-sender chat-sender-ai">IA For HEALTH</span>'
                f'<div class="chat-text">{content}</div>'
                f'</div></div>'
            )
        else:
            html_parts.append(
                f'<div class="chat-row chat-row-user">'
                f'<div class="chat-bubble chat-bubble-user">'
                f'<span class="chat-sender chat-sender-user">Você</span>'
                f'<div class="chat-text">{content}</div>'
                f'</div>'
                f'{user_avatar}'
                f'</div>'
            )
    if html_parts:
        st.markdown(
            '<div class="chat-feed">' + "".join(html_parts) + '</div>',
            unsafe_allow_html=True
        )


def render_multiple_choice_buttons():
    if not st.session_state.started or st.session_state.phase != "quiz" or st.session_state.quiz_completed:
        return
    idx = st.session_state.index
    if idx >= len(st.session_state.questions):
        return
    current_q = st.session_state.questions[idx]
    fase_num = idx + 1
    total = len(st.session_state.questions)

    if current_q["type"] == "open":
        # Pergunta aberta: mostra indicador de fase + texto + dica
        st.markdown(
            f'<div class="fase-indicador">fase {fase_num} de {total}</div>'
            f'<div class="pergunta-destaque">{current_q["question"]}</div>'
            f'<div class="open-question-hint">✍️ Digite sua resposta no campo abaixo e pressione Enter</div>',
            unsafe_allow_html=True
        )
        return

    # Múltipla escolha
    st.markdown(
        f'<div class="fase-indicador">fase {fase_num} de {total}</div>'
        f'<div class="pergunta-destaque">{current_q["question"]}</div>',
        unsafe_allow_html=True
    )
    for i, option in enumerate(current_q["options"]):
        if st.button(option, key=f"mc_{idx}_{i}", use_container_width=True):
            process_multiple_choice(i)
            st.rerun()


def render_unidade_buttons():
    if not (st.session_state.started and st.session_state.phase == "unidade"):
        return
    st.markdown('<div class="result-section-title">🏥 Selecione sua unidade de negócio</div>', unsafe_allow_html=True)
    col1, _ = st.columns([2, 6])
    with col1:
        unidades = [
            ("🏥","Santa Casa BH"),("🏥","São Lucas"),("🧩","Centro de Autismo"),
            ("⚰️","Funerária e Assistência Familiar"),("🎓","Faculdade de Saúde"),
            ("🩺","Ambulatórios Especializados"),("👴","Instituto Geriátrico"),
            ("👶","Instituto Materno Infantil"),("🎗️","Instituto de Oncologia"),
            ("🔬","Pesquisa Clínica"),("🧪","Órix Lab"),("🏢","Corporativo"),
        ]
        for emoji, nome in unidades:
            if st.button(f"{emoji} {nome}", key=f"btn_unidade_{nome.lower().replace(' ','_').replace('(','').replace(')','').replace('ã','a').replace('é','e').replace('ó','o')}"):
                process_unidade(nome); st.rerun()


def render_confirmar_quiz_buttons():
    if not (st.session_state.started and st.session_state.phase == "confirmar_quiz"):
        return
    st.markdown('<div class="result-section-title">🎮 Bora jogar?</div>', unsafe_allow_html=True)
    col1, col2, *_ = st.columns([1, 1, 6])
    with col1:
        if st.button("✅ Sim, bora!", key="btn_confirmar_quiz_sim"):
            st.session_state.aceitou_responder_quiz = "Sim"
            add_msg("user","Sim, bora!",section="quiz")
            mc_qs = [q for q in QUESTION_BANK if q["type"] == "multiple_choice"]
            open_qs = [q for q in QUESTION_BANK if q["type"] == "open"]
            st.session_state.questions = random.sample(mc_qs, min(4, len(mc_qs))) + random.sample(open_qs, min(1, len(open_qs)))
            st.session_state.index = 0; st.session_state.results = []
            st.session_state.phase = "quiz"; st.session_state.focus_input = False
            st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1
            first_q = st.session_state.questions[0]
            add_msg("assistant",
                "🎮 Boa! Vamos começar a aventura! São 5 fases no total — 4 de escolha e 1 aberta! Bora lá! 💜",
                section="quiz")
            st.rerun()
    with col2:
        if st.button("😌 Agora não", key="btn_confirmar_quiz_nao"):
            st.session_state.aceitou_responder_quiz = "Não"
            add_msg("user","Agora não",section="quiz")
            st.session_state.score = 0
            st.session_state.quiz_completed = True
            st.session_state.extra_question_needed = False
            st.session_state.extra_question_answered = True
            st.session_state.fase_papel = False
            st.session_state.phase = "chat"
            st.session_state.focus_input = True
            st.session_state.test_end_time = datetime.now()
            st.session_state.scroll_to_bottom = True; st.session_state.scroll_nonce += 1
            add_msg("assistant",
                f"Tudo bem, {st.session_state.name}! 😊 Sem pressão nenhuma! "
                f"Sua participação foi registrada. "
                f"Lembro que o IA For Health é seu amigo e tá sempre aqui pra tirar suas dúvidas sobre IA! 💜 "
                f"Pode perguntar à vontade 👇",
                section="quiz")
            save_result_to_file(); st.rerun()


def render_papel_jornada_buttons():
    """Botões Instrutor / Aluno / Não — só aparecem quando score >= 4"""
    if not (
        st.session_state.quiz_completed
        and st.session_state.get("fase_papel")
        and st.session_state.phase == "mostrar_resultado"
        and not st.session_state.extra_question_answered
    ):
        return
    st.markdown(
        '<div class="result-section-title">🌟 Você arrasou! Quer fazer parte da nossa jornada de IA?</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="papel-descricao">Escolha como você quer participar da nossa comunidade de aprendizado 😊</div>',
        unsafe_allow_html=True
    )
    col1, col2, col3, *_ = st.columns([1.2, 1.2, 1.2, 4])
    with col1:
        if st.button("🏫 Quero ser Instrutor", key="btn_papel_instrutor", use_container_width=True):
            responder_papel_jornada("Instrutor")
    with col2:
        if st.button("🎓 Quero ser Aluno", key="btn_papel_aluno", use_container_width=True):
            responder_papel_jornada("Aluno")
    with col3:
        if st.button("😌 Não desta vez", key="btn_papel_nao", use_container_width=True):
            responder_papel_jornada("Não")


def render_tema_compartilhamento_form():
    if st.session_state.phase != "tema_compartilhamento":
        return
    st.markdown('<div class="result-section-title">📚 Quais temas você quer explorar?</div>', unsafe_allow_html=True)
    temas = st.multiselect(
        "Selecione um ou mais temas:",
        TEMAS_COMPARTILHAMENTO,
        key="input_tema_compartilhamento",
        placeholder="Clique para escolher os temas..."
    )
    if st.button("💾 Salvar temas", key="btn_salvar_tema_compartilhamento"):
        if not temas:
            st.warning("Selecione pelo menos um tema antes de salvar! 😊"); return
        process_tema_compartilhamento(temas)


def render_final_result():
    if not st.session_state.quiz_completed:
        return
    st.markdown('<div id="card-resultado-avaliacao"></div><div class="result-section-title" style="display:block;">🏆 Resumo da sua jornada</div>', unsafe_allow_html=True)

    # Monta linha extra: Participação (esq) + Temas (dir) em 2 colunas
    papel_html = ""
    if st.session_state.extra_question_answered and st.session_state.papel_jornada:
        temas_val = st.session_state.extra_question_topic or "—"
        papel_html = f'''<div class="final-result-extra-row">
  <div class="final-result-extra-item"><span class="final-result-info-label">Participação na jornada</span><span class="final-result-info-value">{st.session_state.papel_jornada}</span></div>
  <div class="final-result-extra-item"><span class="final-result-info-label">Temas de interesse</span><span class="final-result-info-value">{temas_val}</span></div>
</div>'''

    st.markdown(f"""<div class="final-result-card">
<div class="final-result-top">
  <div><div class="final-result-name">{st.session_state.name}</div><div class="final-result-subtitle">🎮 Jornada de IA — Santa Casa BH</div></div>
  <div class="final-result-score-box"><div class="final-result-score-label">Pontuação</div><div class="final-result-score-value">{st.session_state.score}/5</div></div>
</div>
<div class="final-result-info-grid-fixed">
  <div class="final-result-info-item"><span class="final-result-info-label">Matrícula</span><span class="final-result-info-value">{st.session_state.matricula}</span></div>
  <div class="final-result-info-item"><span class="final-result-info-label">Unidade de negócio</span><span class="final-result-info-value">{st.session_state.unidade_negocio}</span></div>
  <div class="final-result-info-item"><span class="final-result-info-label">Início</span><span class="final-result-info-value">{format_datetime(st.session_state.test_start_time)}</span></div>
</div>
{papel_html}
</div>""", unsafe_allow_html=True)
    with st.container():
        if st.button("🔄 Reiniciar jornada", use_container_width=True, key="btn_reiniciar_dentro_card"):
            reiniciar_teste()
    st.markdown('<div class="result-section-title">📝 Revisão das fases</div>', unsafe_allow_html=True)
    for i, item in enumerate(st.session_state.results, 1):
        status_class = "result-ok" if item["correct"] else "result-error"
        status_text = "✅ Acertou!" if item["correct"] else "💪 Quase lá!"
        badge = '<span class="question-type-badge mc-badge">Múltipla Escolha</span>' if item.get("type") == "multiple_choice" else '<span class="question-type-badge open-badge">Fase Aberta</span>'
        st.markdown(f"""<div class="question-card">
<div style="display:flex;gap:8px;align-items:center;margin-bottom:14px;"><div class="question-status {status_class}">{status_text}</div>{badge}<span style="font-size:.8rem;color:#94a3b8;">Fase {i}</span></div>
<div class="question-title">{item['question']}</div>
<div class="question-block"><div class="question-label">Sua resposta</div><div class="question-text">{item['user_answer']}</div></div>
<div class="question-block"><div class="question-label">Resposta de referência</div><div class="question-text">{item['reference']}</div></div>
<div class="question-block"><div class="question-label">💬 Comentário</div><div class="question-text">{item['feedback']}</div></div>
</div>""", unsafe_allow_html=True)


def render_chat_actions():
    if not st.session_state.quiz_completed:
        return
    st.markdown('<div class="chat-actions-wrap"><a class="btn-ver-resultado-fixo" href="#card-resultado-avaliacao">📊 Ver resumo da jornada</a></div>', unsafe_allow_html=True)
    if st.session_state.phase == "chat":
        st.markdown(
            '<div class="chat-location-card">🤖 <strong>Chat com IA For Health</strong><br>'
            'Seu amigo digital está aqui! Use o campo abaixo pra tirar qualquer dúvida sobre Inteligência Artificial. '
            'Pode perguntar à vontade — sem julgamento, só aprendizado! 💜</div>',
            unsafe_allow_html=True
        )


def render_card_explicacao_chat():
    if st.session_state.get("started", False):
        return
    st.markdown('''<div class="home-grid">
<div class="home-card">🤖 <strong>Sobre a IA For Health</strong><br>
A <strong>IA For Health</strong> é seu companheiro digital na Santa Casa BH para explorar e aprender sobre Inteligência Artificial de um jeito leve e divertido! 💜</div>
<div class="home-card">🎮 <strong>Como funciona a jornada</strong>
<div class="como-funciona-grid compacto">
<div class="como-item"><strong>🙋 Fase de apresentação</strong><br>Conta pra gente quem você é.<br><br>
<strong>🕹️ 5 fases de exploração</strong><br>4 fases de escolha + 1 fase aberta.<br><br>
<strong>🏆 Resumo</strong><br>Veja como foi sua aventura!</div>
<div class="como-item"><strong>💬 Chat livre</strong><br>Converse com a IA e tire dúvidas:<br>
&nbsp;&nbsp;- Conceitos de IA<br>&nbsp;&nbsp;- Exemplos práticos na saúde<br>&nbsp;&nbsp;- Conteúdos internos<br>&nbsp;&nbsp;- O que quiser! 😊</div>
</div></div></div>''', unsafe_allow_html=True)


def render_chat_messages():
    quiz_messages = [m for m in st.session_state.chat if m.get("section","quiz") == "quiz"]
    post_result_messages = [m for m in st.session_state.chat if m.get("section") == "post_result"]
    if quiz_messages:
        render_chat_block(quiz_messages)
        render_unidade_buttons()
    if st.session_state.quiz_completed:
        render_final_result()
    render_papel_jornada_buttons()
    render_tema_compartilhamento_form()
    if st.session_state.quiz_completed:
        render_chat_actions()
    if st.session_state.phase == "chat":
        st.markdown('<div class="result-section-title">💬 Chat com IA For Health</div>', unsafe_allow_html=True)
        if post_result_messages:
            render_chat_block(post_result_messages)
    st.markdown('<div id="fim-da-pagina" style="height:4px;width:100%;display:block;scroll-margin-bottom:90px;"></div>', unsafe_allow_html=True)
    if st.session_state.get("pending_scroll_bottom"):
        st.session_state.scroll_to_bottom = True
        st.session_state.pending_scroll_bottom = False


def injetar_favicon_coracao():
    components.html("""<script>
    (function(){
        var svg = encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 110">'
          + '<path d="M60 98 C60 98 8 62 8 32 C8 16 20 6 34 10 C42 12 50 19 60 28 '
          + 'C70 19 78 12 86 10 C100 6 112 16 112 32 C112 62 60 98 60 98Z" '
          + 'fill="#6300AB"/>'
          + '</svg>'
        );
        var url = 'data:image/svg+xml,' + svg;
        var doc = window.parent.document;
        function setFavicon() {
            doc.querySelectorAll('link[rel*="icon"]').forEach(function(el){ el.remove(); });
            var a = doc.createElement('link');
            a.rel = 'icon'; a.type = 'image/svg+xml'; a.href = url;
            doc.head.appendChild(a);
            var b = doc.createElement('link');
            b.rel = 'shortcut icon'; b.href = url;
            doc.head.appendChild(b);
        }
        setFavicon();
        setTimeout(setFavicon, 800);
        setTimeout(setFavicon, 2500);
    })();
    </script>""", height=0)


def bloquear_tradutor_google():
    components.html("""<script>
    const doc=window.parent.document;
    doc.documentElement.setAttribute("lang","pt-BR");
    doc.documentElement.setAttribute("translate","no");
    doc.body.setAttribute("translate","no");
    doc.body.classList.add("notranslate");
    let meta=doc.querySelector('meta[name="google"]');
    if(!meta){meta=doc.createElement("meta");meta.name="google";meta.content="notranslate";doc.head.appendChild(meta);}
    </script>""", height=0)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(layout="wide", page_title=APP_TITLE, page_icon="💜")

    st.markdown("""
    <style>
    [data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none;}
    .stApp{background:#f7f9fc;}
    html,body,.stApp{-webkit-font-smoothing:antialiased;color:#0f172a!important;background-color:#f7f9fc!important;}
    [data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"],[data-testid="stChatMessage"] p{color:#0f172a!important;opacity:1!important;-webkit-text-fill-color:#0f172a!important;}
    iframe{color-scheme:light;}
    #card-resultado-avaliacao{scroll-margin-top:90px;}

    /* ── Indicador de fase (discreto, não negrito) ── */
    .fase-indicador{
        font-size:.82rem;font-weight:400;color:#6300AB;
        letter-spacing:.3px;margin-bottom:6px;margin-top:16px;
    }
    /* ── Pergunta em destaque (negrito) ── */
    .pergunta-destaque{
        font-size:1.08rem;font-weight:800;color:#0f172a;
        margin-bottom:10px;line-height:1.45;
    }
    /* ── Dica pergunta aberta ── */
    .open-question-hint{
        font-size:.85rem;color:#6300AB;font-weight:500;
        background:#faf5ff;border:1px dashed #d8b4fe;
        border-radius:10px;padding:8px 12px;margin-bottom:14px;
    }

    /* ── Botão ver resultado ── */
    .btn-ver-resultado-fixo{display:block;width:100%;text-align:center;background:#6300AB!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;padding:.75rem 1rem;border-radius:12px;font-weight:800;text-decoration:none!important;margin:.7rem 0;}
    .btn-ver-resultado-fixo *{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;}

    /* ── Títulos de seção ── */
    .result-section-title{font-size:1.15rem;font-weight:800;color:#0f172a;margin-top:24px;margin-bottom:12px;letter-spacing:.2px;}

    /* ── Papel jornada ── */
    .papel-descricao{font-size:.92rem;color:#5b21b6;margin-bottom:14px;background:#faf5ff;border:1px solid #e9d5ff;border-radius:10px;padding:10px 14px;}

    /* ── Cards resultado ── */
    .final-result-card{background:linear-gradient(180deg,#fff 0%,#faf5ff 100%);border:1px solid #d8b4fe;border-left:6px solid #6300AB;border-radius:16px;padding:20px 22px;box-shadow:0 10px 24px rgba(99,0,171,.08);margin-bottom:12px;}
    .final-result-top{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:16px;flex-wrap:wrap;padding-bottom:12px;border-bottom:1px solid #ede9fe;}
    .final-result-name{font-size:1.5rem;font-weight:800;color:#0f172a;line-height:1.2;margin-bottom:3px;}
    .final-result-subtitle{font-size:.9rem;color:#64748b;font-weight:500;}
    .final-result-score-box{background:linear-gradient(135deg,#f3e8ff 0%,#ede9fe 100%);border:1px solid #d8b4fe;border-radius:14px;padding:12px 16px;min-width:130px;text-align:center;}
    .final-result-score-label{font-size:.75rem;color:#475569;font-weight:700;margin-bottom:4px;text-transform:uppercase;letter-spacing:.4px;}
    .final-result-score-value{font-size:1.5rem;font-weight:800;color:#6300AB;line-height:1.1;}
    .final-result-info-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;}
    /* Grid fixo 3 colunas para matrícula/unidade/início */
    .final-result-info-grid-fixed{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px;}
    .final-result-info-item{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px 14px;}
    .final-result-info-item-full{grid-column:1/-1;background:#faf5ff;}
    /* Cards extras (papel/temas) em 2 colunas */
    .final-result-extra-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;}
    .final-result-extra-item{background:#faf5ff;border:1px solid #d8b4fe;border-radius:12px;padding:12px 14px;}
    @media(max-width:600px){
        .final-result-info-grid-fixed{grid-template-columns:1fr 1fr;}
        .final-result-extra-row{grid-template-columns:1fr;}
    }
    .final-result-info-label{display:block;font-size:.74rem;font-weight:800;color:#64748b;margin-bottom:5px;text-transform:uppercase;letter-spacing:.45px;}
    .final-result-info-value{display:block;font-size:.98rem;font-weight:600;color:#0f172a;line-height:1.35;}

    /* ── Questão card ── */
    .question-card{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px;margin-bottom:14px;box-shadow:0 4px 14px rgba(15,23,42,.04);}
    .question-status{display:inline-block;padding:6px 12px;border-radius:999px;font-size:.85rem;font-weight:700;}
    .result-ok{background:#ecfdf3;color:#166534;border:1px solid #bbf7d0;}
    .result-error{background:#fff7ed;color:#92400e;border:1px solid #fde68a;}
    .question-title{font-size:1.08rem;font-weight:700;color:#0f172a;margin-bottom:16px;}
    .question-block{margin-bottom:14px;}
    .question-label{font-size:.9rem;font-weight:700;color:#475569;margin-bottom:6px;}
    .question-text{font-size:.98rem;color:#334155;line-height:1.55;background:#f8fafc;border-radius:12px;padding:12px 14px;border:1px solid #e2e8f0;}
    .question-type-badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:.78rem;font-weight:700;}
    .mc-badge{background:#f3e8ff;color:#6300AB;border:1px solid #d8b4fe;}
    .open-badge{background:#ecfdf3;color:#166534;border:1px solid #86efac;}

    /* ── Multiselect tema — toque roxo ── */
    [data-baseweb="select"] > div,[data-baseweb="select"] span{border-color:#6300AB!important;color:#6300AB!important;}
    [data-testid="stMultiSelect"] [data-baseweb="tag"]{background:#f3e8ff!important;color:#6300AB!important;border:1px solid #d8b4fe!important;}
    [data-testid="stMultiSelect"] [data-baseweb="tag"] span{color:#6300AB!important;-webkit-text-fill-color:#6300AB!important;}

    .chat-actions-wrap{margin-top:8px;margin-bottom:16px;}
    .chat-location-card{background:#faf5ff;border:1px dashed #d8b4fe;color:#334155;border-radius:14px;padding:12px 14px;margin-top:6px;margin-bottom:16px;font-size:.95rem;line-height:1.45;}

    /* ── OCULTA ícones padrão do Streamlit ── */
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"],
    [data-testid="stChatMessage"] .stChatMessageAvatarUser,
    [data-testid="stChatMessage"] .stChatMessageAvatarAssistant,
    [data-testid="stChatMessage"] > div:first-child > img,
    [data-testid="stChatMessage"] > div:first-child > svg { display:none!important; }

    /* ── Chat feed ── */
    .chat-feed { display:flex; flex-direction:column; gap:14px; margin:8px 0 16px 0; }
    .chat-row { display:flex; align-items:flex-start; gap:10px; }
    .chat-row-user { flex-direction:row-reverse; }
    .chat-avatar { flex-shrink:0; width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center; }
    .chat-avatar-ai   { background:#f3e8ff; border:2px solid #d8b4fe; }
    .chat-avatar-user { background:#ede9fe; border:2px solid #c4b5fd; }
    .chat-bubble { max-width:78%; border-radius:16px; padding:12px 16px; line-height:1.55; font-size:.97rem; color:#0f172a; word-break:break-word; }
    .chat-bubble-ai { background:#ffffff; border:1px solid #e2e8f0; border-top-left-radius:4px; box-shadow:0 2px 8px rgba(99,0,171,.06); }
    .chat-bubble-user { background:#ffffff; border:2px solid #6300AB; color:#0f172a!important; border-top-right-radius:4px; box-shadow:0 2px 8px rgba(99,0,171,.10); }
    .chat-bubble-user .chat-text { color:#0f172a!important; }
    .chat-sender { display:block; font-size:.75rem; font-weight:800; margin-bottom:5px; letter-spacing:.3px; }
    .chat-sender-ai   { color:#6300AB; }
    .chat-sender-user { color:#6300AB; }
    .chat-text { font-size:.97rem; line-height:1.6; }
    @media(max-width:768px){
        .chat-bubble { max-width:90%; font-size:.93rem; padding:10px 13px; }
        .chat-avatar  { width:34px; height:34px; }
    }

    /* ── Status bar ── */
    .status-bar{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border:1px solid #e2e8f0;border-radius:12px;background:#fff;overflow:hidden;margin-bottom:4px;}
    .status-item{padding:8px 12px;border-right:1px solid #e2e8f0;}
    .status-item:last-child{border-right:none;}
    .status-label{display:block;font-size:.68rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px;}
    .status-value{display:block;font-size:.92rem;font-weight:700;color:#0f172a;word-break:break-word;white-space:normal;overflow-wrap:anywhere;}
    @media(max-width:900px){
        .status-bar{grid-template-columns:repeat(2,1fr);}
        .status-item:nth-child(2n){border-right:none;}
        .status-item:nth-child(3),.status-item:nth-child(4){border-top:1px solid #e2e8f0;}
    }
    @media(max-width:600px){
        .status-bar{grid-template-columns:1fr 1fr;}
        .status-item{border-right:1px solid #e2e8f0;border-top:1px solid #e2e8f0;}
        .status-item:nth-child(1),.status-item:nth-child(2){border-top:none;}
        .status-item:nth-child(2n){border-right:none;}
        .status-value{font-size:.93rem;}
    }

    /* ── Chat input ── */
    div[data-testid="stChatInput"]{border:1px solid #6300AB!important;border-radius:16px!important;background:white!important;margin-bottom:0px!important;position:relative!important;bottom:0!important;}
    div[data-testid="stChatInput"]:focus-within{box-shadow:0 0 0 2px rgba(99,0,171,.2)!important;}
    div[data-testid="stChatInput"] textarea,div[data-testid="stChatInput"] input{border:none!important;outline:none!important;box-shadow:none!important;border-radius:16px!important;}
    div[data-testid="stChatInput"]>div,div[data-testid="stChatInput"]>div>div{border:none!important;box-shadow:none!important;border-radius:16px!important;background:white!important;}

    /* ── Botão enviar → coração roxo ── */
    div[data-testid="stChatInput"] button,
    [data-testid="stChatInputSubmitButton"],
    button[data-testid="stChatInputSubmitButton"]{background:linear-gradient(135deg,#6300AB 0%,#5000A0 100%)!important;border-radius:10px!important;border:none!important;width:36px!important;height:36px!important;padding:0!important;display:flex!important;align-items:center!important;justify-content:center!important;position:relative!important;overflow:hidden!important;flex-shrink:0!important;}
    div[data-testid="stChatInput"] button:hover,[data-testid="stChatInputSubmitButton"]:hover{background:linear-gradient(135deg,#5000A0 0%,#3d0080 100%)!important;transform:scale(1.06)!important;transition:all .15s ease!important;}
    div[data-testid="stChatInput"] button svg,[data-testid="stChatInputSubmitButton"] svg{display:none!important;}
    div[data-testid="stChatInput"] button::after,[data-testid="stChatInputSubmitButton"]::after{content:"";display:block;width:20px;height:20px;background-color:#ffffff;-webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 90'%3E%3Cpath d='M50 85 C50 85 5 55 5 28 C5 14 17 5 30 9 C37 11 44 17 50 25 C56 17 63 11 70 9 C83 5 95 14 95 28 C95 55 50 85 50 85Z'/%3E%3C/svg%3E");-webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 90'%3E%3Cpath d='M50 85 C50 85 5 55 5 28 C5 14 17 5 30 9 C37 11 44 17 50 25 C56 17 63 11 70 9 C83 5 95 14 95 28 C95 55 50 85 50 85Z'/%3E%3C/svg%3E");mask-size:contain;mask-repeat:no-repeat;mask-position:center;}

    /* ── stBottom ── */
    [data-testid="stBottom"]{padding-bottom:2px!important;padding-top:4px!important;background:transparent!important;}

    /* ── Todos os botões → roxo ── */
    .stButton button,div[data-testid="stButton"] button,button[kind="primary"],button[kind="secondary"]{background-color:#6300AB!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;border-radius:12px!important;border:none!important;font-weight:700!important;}
    .stButton button *,div[data-testid="stButton"] button *,div[data-testid="stButton"] button p,div[data-testid="stButton"] button span{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;font-weight:700!important;}

    /* ── Header / layout principal ── */
    [data-testid="stMainBlockContainer"],.block-container{
        padding-top:1rem!important;
        padding-bottom:max(90px,calc(70px + env(safe-area-inset-bottom)))!important;
    }
    .top-header{display:grid;grid-template-columns:1fr 2fr 1fr;align-items:center;margin-top:-10px;margin-bottom:6px;}
    .app-title-center{text-align:center!important;font-size:1.9rem!important;font-weight:900!important;color:#2f3140!important;margin:0!important;line-height:1.1!important;display:flex;align-items:center;justify-content:center;gap:6px;}
    .app-version-right{text-align:right;color:#7b8190;font-size:.82rem;font-weight:500;padding-top:6px;}
    hr{margin-top:4px!important;margin-bottom:6px!important;}

    /* ── Âncora de fim ── */
    #fim-da-pagina{height:4px;width:100%;display:block;scroll-margin-bottom:90px;}

    /* ── iOS overscroll ── */
    html,body{overscroll-behavior-y:contain;-webkit-overflow-scrolling:touch;}
    section.main{scroll-behavior:auto!important;}

    /* ── Start card ── */
    .start-card-shell{background:linear-gradient(90deg,#faf5ff 0%,#f5f3ff 100%);border:1px solid #d8b4fe;border-radius:14px;padding:12px 18px;margin:6px 0 10px 0;box-shadow:0 6px 18px rgba(99,0,171,.07);}
    .start-card-title{font-size:.97rem;font-weight:800;color:#3b0764;margin-bottom:2px;}
    .start-card-subtitle{font-size:.9rem;color:#5b21b6;}

    /* ── Btn primário / secundário ── */
    .btn-primary-wrap button{background:linear-gradient(135deg,#6300AB 0%,#5000A0 100%)!important;color:#ffffff!important;border-radius:14px!important;height:48px!important;font-weight:800!important;border:none!important;box-shadow:0 8px 20px rgba(99,0,171,.28)!important;}
    .btn-secondary-wrap button{background:#fff!important;color:#6300AB!important;-webkit-text-fill-color:#6300AB!important;border:1.5px solid #d8b4fe!important;border-radius:14px!important;height:48px!important;font-weight:800!important;}
    .btn-secondary-wrap button p,.btn-secondary-wrap button span{color:#6300AB!important;-webkit-text-fill-color:#6300AB!important;}

    /* ── Home grid ── */
    .home-grid{display:grid;grid-template-columns:1fr 1.3fr;gap:10px;margin-top:6px;}
    .home-card{background:#faf5ff;border:1px solid #e9d5ff;border-radius:14px;padding:12px 14px;color:#3b0764;font-size:.9rem;line-height:1.35;}
    .como-funciona-grid.compacto{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;}
    .como-item{background:#fff;border:1px solid #e9d5ff;border-left:4px solid #6300AB;border-radius:10px;padding:9px 12px;color:#0f172a;font-size:.85rem;line-height:1.3;}

    @media(max-width:900px){.home-grid,.como-funciona-grid.compacto{grid-template-columns:1fr;}}
    @media(max-width:768px){
        html,body,.stApp{overflow-x:hidden!important;}
        .main .block-container,[data-testid="stMainBlockContainer"]{
            padding-left:1rem!important;padding-right:1rem!important;
            padding-bottom:max(100px,calc(80px + env(safe-area-inset-bottom)))!important;
            max-width:100%!important;
        }
        .final-result-card,.chat-location-card,.question-card,.home-card{max-width:100%!important;overflow-wrap:break-word!important;word-break:break-word!important;}
        [data-testid="stChatInput"]{padding-bottom:env(safe-area-inset-bottom)!important;}
        .top-header{grid-template-columns:1fr;}
        .app-version-right{text-align:center;}
        .app-title-center{font-size:1.5rem!important;}
    }
    </style>""", unsafe_allow_html=True)

    components.html("""<script>
    document.documentElement.setAttribute("translate","no");
    document.body.setAttribute("translate","no");
    </script>""", height=0)

    load_environment()
    init_state()
    render_header()
    render_status()
    render_card_explicacao_chat()
    render_chat_messages()
    render_confirmar_quiz_buttons()
    render_multiple_choice_buttons()
    save_result_to_file()

    if st.session_state.started and st.session_state.phase not in ["extra","unidade","confirmar_quiz","mostrar_resultado"]:
        is_mc_waiting = (
            st.session_state.phase == "quiz"
            and not st.session_state.quiz_completed
            and bool(st.session_state.get("questions"))
            and st.session_state.index < len(st.session_state.questions)
            and st.session_state.questions[st.session_state.index]["type"] == "multiple_choice"
        )
        if not is_mc_waiting:
            user_text = st.chat_input("💬 Digite sua mensagem aqui...")
            if user_text:
                st.markdown(
                    '<div class="chat-feed"><div class="chat-row chat-row-ai">'
                    '<div class="chat-avatar chat-avatar-ai"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 90" width="28" height="28">'
                    '<path d="M50 85 C50 85 5 55 5 28 C5 14 17 5 30 9 C37 11 44 17 50 25 C56 17 63 11 70 9 C83 5 95 14 95 28 C95 55 50 85 50 85Z" fill="#6300AB"/>'
                    '</svg></div>'
                    '<div class="chat-bubble chat-bubble-ai">'
                    '<span class="chat-sender chat-sender-ai">IA For HEALTH</span>'
                    '<div class="chat-text">⏳ Um segundo...</div>'
                    '</div></div></div>',
                    unsafe_allow_html=True
                )
                try:
                    process_user_message(user_text)
                except Exception as exc:
                    add_msg("assistant",f"Ops! 😅 Ocorreu um erro: {exc}",section="post_result")
                st.rerun()

    bloquear_tradutor_google()
    injetar_favicon_coracao()
    render_footer_fixo()
    run_scroll_to_result_script()
    run_focus_script()
    _render_scroll_anchor()


if __name__ == "__main__":
    main()
