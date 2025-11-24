📘 README — Como transformar a base de conhecimento em uma POC de agente GenAI (RAG)

Este guia é opcional e serve para quem deseja transformar o arquivo
base_conhecimento_ifood_genai.csv em uma prova de conceito (POC) de um agente interno usado para decisões de reembolsos e cancelamentos, semelhante ao que times internos podem desenvolver no iFood.

A ideia não é construir um sistema completo, mas criar algo demonstrável para portfólio, currículo ou entrevista técnica.

🎯 Objetivo da POC

Criar um agente de IA capaz de:

Consultar informações oficiais (base de conhecimento)

Responder perguntas operacionais de forma consistente

Evitar respostas inventadas (alucinação)

Sugerir fallback seguro quando não houver confiança suficiente

🧰 O que você vai precisar

Você pode escolher entre duas rotas:

Tipo de POC	Ferramentas sugeridas
Sem código	Flowise, Dify, ChatGPT Assistants, N8n, Zapier AI Actions
Com código	Python + alguma biblioteca de RAG (LangChain, LlamaIndex etc.)

Se seu foco é portfólio rápido, comece com no-code.

📥 1. Importe a base de conhecimento

Faça upload do arquivo CSV na ferramenta escolhida, dentro da seção onde ela aceita:

Knowledge Base

Documents

Files / Upload

Sources / Data Sources

Verifique se o conteúdo foi indexado (algumas plataformas mostram isso).

🧠 2. Configure o propósito do agente

Use uma descrição como esta:

Você é um agente interno que auxilia colaboradores a decidirem sobre reembolsos e cancelamentos.
Sempre consulte a base de conhecimento antes de responder.
Se não houver confiança suficiente, sugira validação manual ou abertura de ticket interno, em vez de responder com incerteza.

🔍 3. Ative o uso da base com busca semântica (RAG)

Procure por opções como:

“Use knowledge in answers”

“Ground responses on documents”

“Retrieval / Semantic Search / RAG”

“Search documents before answering”

Ative e mantenha as configurações padrão.

⚠️ 4. Configure um comportamento de fallback

Sugestão de frase padrão:

Não encontrei informações suficientes na base para responder com segurança. Sugiro abrir um ticket interno ou consultar a política oficial.

🧪 5. Teste com cenários reais

Use perguntas como:

Pergunta recomendada	O que observar
“O cliente quer reembolso, mas o pedido já saiu para entrega. Ainda é permitido?”	O agente deve diferenciar motivos (erro do app/restaurante/entregador vs. desistência do cliente)
“O restaurante cancelou por falta de ingrediente. O reembolso é automático?”	Deve localizar políticas de reembolso automático
“O cliente foi cobrado após o cancelamento. O que fazer?”	Deve mencionar validação do estorno e possível ticket
📝 6. Como apresentar essa POC no portfólio (modelo pronto)

Desenvolvi uma POC de agente interno para decisões de reembolso/cancelamento utilizando RAG com uma base de conhecimento simulada.
Configurei fallback para baixa confiança e testei cenários críticos (pedido já saiu para entrega, cancelamento por falha do restaurante, cobrança após cancelamento).
A POC foi criada com foco em consistência operacional e redução de respostas incorretas.

🚀 7. Ideias de evolução (opcional)

logs de confiança da resposta

categorias de decisão (financeiro / restaurante / entrega / fraude)

integração com APIs fictícias de pedido/estorno

classificação automática do tipo de caso

📂 Arquivo utilizado

base_conhecimento_ifood_genai.csv
