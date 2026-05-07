# ROTEIRO DE INSTALAÇÃO DAS FERRAMENTAS
## MBA RAG & CAG Aplicados a Direito e Segurança Pública — Aula 1

> **Versão:** 2.0 | **Atualizado:** Abril/2026  
> **Sistema Operacional:** Ubuntu 22.04 LTS / macOS 13+ / Windows 11  
> **Python:** 3.11+

---

## ⚡ ÍNDICE RÁPIDO

| # | Ferramenta | Tempo Estimado | Seção |
|---|-----------|----------------|-------|
| 1 | Docker Desktop | 15 min | [→ 1](#1-docker-desktop) |
| 2 | OpenSearch 3.x + Dashboards | 15 min | [→ 2](#2-opensearch-3x-e-opensearch-dashboards) |
| 3 | ngrok — Túnel para Google Colab | 10 min | [→ 3](#3-ngrok--túnel-local-para-google-colab) |
| 4 | Python 3.11 + pip | 5 min | [→ 4](#4-python-311-e-pip) |
| 5 | Ambiente Virtual + Dependências | 10 min | [→ 5](#5-ambiente-virtual-e-dependências-python) |
| 6 | vLLM — Servidor de Inferência LLM | 25 min | [→ 6](#6-vllm--servidor-de-inferência-llm) |
| 7 | LangFuse | 10 min | [→ 7](#7-langfuse) |
| 8 | Variáveis de Ambiente | 5 min | [→ 8](#8-variáveis-de-ambiente) |
| 9 | Validação Final | 10 min | [→ 9](#9-validação-final-do-ambiente) |
| 10 | Google Colab (alternativa) | 5 min | [→ 10](#10-configuração-google-colab-alternativa-cloud) |

**Tempo total estimado:** ~105 minutos (primeira vez) | ~20 minutos (sessões posteriores)

---

## PRÉ-REQUISITOS DE HARDWARE

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| RAM | 16 GB | 32 GB |
| GPU VRAM | 8 GB (NVIDIA, para vLLM) | 16 GB (NVIDIA T4/RTX 3080+) |
| Armazenamento | 80 GB livre | 150 GB livre |
| CPU | 4 cores | 8+ cores |
| Conexão | 10 Mbps | 100+ Mbps |

> **📌 Sem GPU local?** Use o Google Colab (seção 10) para o vLLM. Para o OpenSearch, instale localmente via Docker Desktop e exponha via ngrok (seção 3).

---

## 1. Docker Desktop

O **Docker Desktop** é a solução unificada para rodar contêineres em todos os sistemas operacionais — Ubuntu, macOS e Windows. Ele inclui Docker Engine, Docker Compose e uma interface gráfica integrada.

> **Por que Docker Desktop e não Docker Engine no Windows?**  
> O Docker Desktop no Windows gerencia o WSL2 internamente e não requer configurações manuais de integração. Para o curso, isso simplifica a execução do OpenSearch e facilita a instalação do plugin ngrok diretamente na interface gráfica.

### 1.1 Instalação no Ubuntu 22.04 LTS

No Ubuntu, o Docker Desktop pode ser instalado via pacote `.deb` oficial:

```bash
# Passo 1: Instalar dependências
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Passo 2: Adicionar chave GPG do Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Passo 3: Adicionar repositório Docker
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Passo 4: Baixar pacote do Docker Desktop para Ubuntu
# Baixe manualmente em: https://docs.docker.com/desktop/install/ubuntu/
# Ou use wget com a URL mais recente:
DOCKER_DESKTOP_URL="https://desktop.docker.com/linux/main/amd64/docker-desktop-amd64.deb"
wget -O /tmp/docker-desktop.deb "$DOCKER_DESKTOP_URL"

# Passo 5: Instalar o pacote
sudo apt-get install -y /tmp/docker-desktop.deb

# Passo 6: Iniciar o Docker Desktop
systemctl --user start docker-desktop

# Passo 7: Habilitar início automático
systemctl --user enable docker-desktop

# Passo 8: Verificar versões
docker --version
docker compose version
```

**Saída esperada:**
```
Docker version 27.x.x, build xxxxxxx
Docker Compose version v2.x.x
```

> **Alternativa CLI-only no Ubuntu:** Se preferir apenas Docker Engine (sem interface gráfica), siga:
> ```bash
> sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
> sudo usermod -aG docker $USER && newgrp docker
> ```

### 1.2 Instalação no macOS (Homebrew)

```bash
# Passo 1: Instalar Docker Desktop via Homebrew Cask
brew install --cask docker

# Passo 2: Abrir o Docker Desktop pela primeira vez
open /Applications/Docker.app

# Passo 3: Aguardar o ícone da baleia 🐳 aparecer na barra de menu
#          (pode levar 30-60 segundos na primeira vez)

# Passo 4: Verificar
docker --version
docker compose version
```

> **Primeiro acesso:** O macOS pode pedir permissão de acessibilidade. Aceite e aguarde o Docker Desktop completar a inicialização do VM interno.

### 1.3 Instalação no Windows 11 (Docker Desktop — sem WSL2 manual)

O Docker Desktop para Windows gerencia o WSL2 automaticamente. **Não é necessário instalar ou configurar o WSL2 manualmente**.

```
Passo 1: Verificar pré-requisitos do sistema
  - Windows 11 64-bit (versão 22H2 ou superior)
  - Virtualização habilitada na BIOS/UEFI
  - Verificar: Gerenciador de Tarefas → Desempenho → CPU → Virtualização: "Habilitado"

Passo 2: Baixar Docker Desktop
  - Acesse: https://docs.docker.com/desktop/install/windows-install/
  - Clique em "Docker Desktop for Windows"
  - Salve o arquivo Docker Desktop Installer.exe

Passo 3: Executar o instalador
  - Dê duplo clique em "Docker Desktop Installer.exe"
  - Mantenha marcado: ✅ "Use WSL 2 instead of Hyper-V"
  - Mantenha marcado: ✅ "Add shortcut to desktop"
  - Clique em "Ok" e aguarde a instalação

Passo 4: Reiniciar o computador
  - O instalador solicitará reinicialização — aceite

Passo 5: Primeiro acesso
  - Abra o Docker Desktop pela área de trabalho
  - Aceite os termos de uso
  - Aguarde a inicialização (ícone da baleia na barra de tarefas)

Passo 6: Verificar no PowerShell (ou Terminal do Windows)
```

```powershell
# Execute no PowerShell ou Windows Terminal
docker --version
docker compose version

# Resultado esperado:
# Docker version 27.x.x, build xxxxxxx
# Docker Compose version v2.x.x
```

> **⚠️ Não configure WSL2 manualmente.** O Docker Desktop instala e gerencia sua própria distribuição WSL2 (`docker-desktop`). Configurações manuais de WSL2 podem conflitar com o Docker Desktop.

### 1.4 Configuração de Memória para OpenSearch (Ubuntu e WSL2 interno)

> **⚠️ OBRIGATÓRIO no Ubuntu.** No macOS e Windows (Docker Desktop), esta configuração **não é necessária** — o Docker Desktop gerencia os parâmetros do kernel do VM interno.

```bash
# Ubuntu: Configura limite de mapa de memória virtual do kernel
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
sudo sysctl -w vm.max_map_count=262144

# Tornar permanente após reinicialização:
echo 'vm.max_map_count=262144' | sudo tee /etc/sysctl.d/99-opensearch.conf

# Verificar
sysctl vm.max_map_count
# Deve retornar: vm.max_map_count = 262144
```

### 1.5 Comandos Úteis do Docker

```bash
# Verificar contêineres em execução
docker ps

# Verificar uso de recursos
docker stats --no-stream

# Ver logs de um contêiner
docker logs -f opensearch-node1

# Parar todos os serviços de um compose
docker compose stop

# Remover contêineres (preserva volumes)
docker compose down

# Remover contêineres E volumes (PERDE DADOS)
docker compose down -v
```

---

## 2. OpenSearch 3.x e OpenSearch Dashboards

O OpenSearch será o motor de busca híbrida (vetorial + textual) do curso. Rodará via Docker Desktop localmente e será acessível pelo Google Colab através do ngrok (seção 3).

### 2.1 Criar o Diretório e o docker-compose.yml

**Ubuntu / macOS (terminal):**
```bash
mkdir -p ~/mba-rag/infra
cd ~/mba-rag/infra

cat > docker-compose.yml << 'EOF'
version: '3'
services:
  opensearch-node1:
    image: opensearchproject/opensearch:3.0.0
    container_name: opensearch-node1
    environment:
      - cluster.name=opensearch-cluster-rag
      - node.name=opensearch-node1
      - discovery.seed_hosts=opensearch-node1
      - cluster.initial_cluster_manager_nodes=opensearch-node1
      - bootstrap.memory_lock=true
      - OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g
      # ATENÇÃO: Segurança desabilitada apenas para desenvolvimento
      # Em produção: configure TLS, certificados e autenticação
      - plugins.security.disabled=true
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536
    volumes:
      - opensearch-data1:/usr/share/opensearch/data
    ports:
      - 9200:9200
      - 9600:9600
    networks:
      - rag-network
    healthcheck:
      test: ["CMD-SHELL", "curl -s http://localhost:9200/_cluster/health | grep -qE '\"status\":\"(green|yellow)\"'"]
      interval: 30s
      timeout: 10s
      retries: 5

  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:3.0.0
    container_name: opensearch-dashboards
    ports:
      - 5601:5601
    environment:
      OPENSEARCH_HOSTS: '["http://opensearch-node1:9200"]'
      DISABLE_SECURITY_DASHBOARDS_PLUGIN: "true"
    depends_on:
      opensearch-node1:
        condition: service_healthy
    networks:
      - rag-network

volumes:
  opensearch-data1:

networks:
  rag-network:
    driver: bridge
EOF

echo "✅ docker-compose.yml criado!"
```

**Windows (PowerShell):**
```powershell
# Cria diretório
New-Item -ItemType Directory -Force -Path "$HOME\mba-rag\infra"
Set-Location "$HOME\mba-rag\infra"

# Cria o docker-compose.yml
# Copie o conteúdo acima e salve como docker-compose.yml
# Ou use o Bloco de Notas: notepad docker-compose.yml
```

### 2.2 Iniciar o OpenSearch

```bash
# Ubuntu / macOS
cd ~/mba-rag/infra
docker compose up -d

# Aguardar inicialização (60-90 segundos na primeira vez)
echo "⏳ Aguardando OpenSearch inicializar..."
sleep 75

# Verificar status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Verificar saúde do cluster
curl -s http://localhost:9200/_cluster/health | python3 -m json.tool
```

**Windows (PowerShell):**
```powershell
Set-Location "$HOME\mba-rag\infra"
docker compose up -d

Start-Sleep -Seconds 75

docker ps --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"

# Verificar cluster (requer curl ou Invoke-RestMethod)
Invoke-RestMethod -Uri "http://localhost:9200/_cluster/health" | ConvertTo-Json
```

**Saída esperada do `_cluster/health`:**
```json
{
  "cluster_name": "opensearch-cluster-rag",
  "status": "yellow",
  "number_of_nodes": 1,
  "active_shards": 0
}
```
> Status "yellow" é normal em single-node (réplicas não alocadas).

### 2.3 Verificar OpenSearch Dashboards

Acesse no navegador: **http://localhost:5601**

```bash
# Ou verifica via curl:
curl -s -o /dev/null -w "%{http_code}" http://localhost:5601
# Deve retornar: 200
```

---

## 3. ngrok — Túnel Local para Google Colab

O **ngrok** cria um túnel HTTPS seguro do seu computador local para a internet, permitindo que o Google Colab acesse o OpenSearch (e o vLLM) rodando na sua máquina.

```
Sua Máquina Local
  └── Docker Desktop → opensearch-node1:9200
                      ↑
                   ngrok cria túnel
                      ↓
              https://abc123.ngrok-free.app  ← URL pública HTTPS
                      ↑
              Google Colab acessa via HTTPS
```

### 3.1 Criar Conta e Obter Token ngrok

```
1. Acesse: https://ngrok.com
2. Clique em "Sign up" — crie conta gratuita (email ou GitHub)
3. Após login, vá em: Dashboard → Your Authtoken
4. Copie o token (formato: 2abc...xxxxx_yyyyyyy)
5. Guarde — você precisará dele nas seções 3.2 e 3.3
```

### 3.2 Método A — Plugin ngrok no Docker Desktop (Recomendado)

O Docker Desktop suporta extensões/plugins. O plugin ngrok cria túneis diretamente pela interface gráfica — sem necessidade de instalar o ngrok separadamente.

#### Instalação do Plugin (todos os SOs):

```
1. Abra o Docker Desktop
2. Clique em "Extensions" (ou "Extensões") no menu lateral esquerdo
3. Na barra de busca, digite: ngrok
4. Clique em "ngrok" nos resultados e clique em "Install"
5. Aguarde a instalação (30-60 segundos)
6. O plugin ngrok aparecerá no menu lateral do Docker Desktop
```

#### Configurar o Authtoken:

```
1. No Docker Desktop, clique em "ngrok" no menu lateral
2. Na tela de configuração, cole seu Authtoken no campo "Auth Token"
3. Clique em "Save" ou "Connect"
```

#### Criar o Túnel para o OpenSearch:

```
1. Na interface do plugin ngrok, clique em "Create Tunnel" (ou "New Tunnel")
2. Configure:
   - Protocol: HTTP
   - Port: 9200
   - (Opcional) Domain: deixe em branco para URL aleatória
3. Clique em "Start Tunnel"
4. Uma URL HTTPS será gerada, exemplo:
   https://abc123def456.ngrok-free.app
5. Copie esta URL — você usará nos notebooks do Colab
```

> **⚠️ Conta Gratuita:** A URL muda a cada vez que o túnel é recriado. Para URL fixa, use o plano pago ou salve a URL antes de fechar.

### 3.3 Método B — ngrok via CLI (Ubuntu, macOS, Windows)

Para quem prefere linha de comando ou não usa Docker Desktop.

#### Ubuntu:
```bash
# Passo 1: Baixar e instalar ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
    sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
    sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update
sudo apt install ngrok

# Passo 2: Autenticar com seu token
ngrok config add-authtoken SEU_TOKEN_AQUI

# Passo 3: Verificar instalação
ngrok version
```

#### macOS:
```bash
# Via Homebrew
brew install ngrok/ngrok/ngrok

# Autenticar
ngrok config add-authtoken SEU_TOKEN_AQUI

# Verificar
ngrok version
```

#### Windows (PowerShell):
```powershell
# Via Chocolatey (se instalado)
choco install ngrok

# Ou baixe o executável em: https://ngrok.com/download
# Extraia ngrok.exe para C:\ngrok\

# Autenticar (no PowerShell)
ngrok config add-authtoken SEU_TOKEN_AQUI

# Verificar
ngrok version
```

#### Criar o Túnel via CLI:
```bash
# Ubuntu / macOS — Inicia o túnel para OpenSearch (porta 9200)
ngrok http 9200 &

# A saída exibirá algo como:
# Forwarding  https://abc123def.ngrok-free.app -> http://localhost:9200

# Para ver o túnel ativo:
curl http://localhost:4040/api/tunnels | python3 -m json.tool
# O campo "public_url" tem a URL HTTPS para usar no Colab
```

```powershell
# Windows PowerShell — Inicia em nova janela
Start-Process ngrok -ArgumentList "http 9200"
# Ou em background:
Start-Job -ScriptBlock { ngrok http 9200 }

# Obter URL do túnel ativo
Invoke-RestMethod -Uri "http://localhost:4040/api/tunnels" | ConvertTo-Json -Depth 5
```

### 3.4 Testar a URL do ngrok

Após criar o túnel, teste a conectividade:

```bash
# Substitua pela sua URL ngrok
NGROK_URL="https://abc123def456.ngrok-free.app"

# Testa o endpoint do OpenSearch via ngrok
curl -s "$NGROK_URL/_cluster/health" \
     -H "ngrok-skip-browser-warning: true" | python3 -m json.tool
```

**Saída esperada:**
```json
{
  "cluster_name": "opensearch-cluster-rag",
  "status": "yellow",
  "number_of_nodes": 1
}
```

> **Por que o header `ngrok-skip-browser-warning`?**  
> Contas gratuitas do ngrok exibem uma página de aviso no navegador para URLs HTTP. O header `ngrok-skip-browser-warning: true` instrui o ngrok a retornar o conteúdo diretamente, sem a página de aviso. Este header deve ser incluído em **todas as requisições** feitas pelos notebooks.

### 3.5 Boas Práticas de Segurança com ngrok

```
⚠️  O que FAZER:
  ✅ Use ngrok apenas para desenvolvimento e testes
  ✅ Encerre o túnel quando não estiver usando (docker compose stop / Ctrl+C)
  ✅ Use apenas com dados fictícios ou anonimizados
  ✅ Prefira o plano pago com IP whitelist para ambientes sensíveis

❌  O que NÃO FAZER:
  ❌ Nunca exponha via ngrok dados reais de processos judiciais
  ❌ Nunca deixe o túnel ativo sem necessidade
  ❌ Nunca compartilhe a URL ngrok publicamente
  ❌ Nunca use ngrok como solução de produção (use VPN ou rede privada)
```

---

## 4. Python 3.11 e pip

### 4.1 Verificar Python Existente

```bash
python3 --version   # Deve retornar 3.11.x ou superior
pip3 --version
```

### 4.2 Instalar Python 3.11 (Ubuntu)

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-pip

python3.11 --version
```

### 4.3 Instalar Python 3.11 (macOS)

```bash
brew install python@3.11

python3.11 --version
```

### 4.4 Instalar Python 3.11 (Windows)

```
1. Acesse: https://www.python.org/downloads/windows/
2. Baixe "Python 3.11.x — Windows installer (64-bit)"
3. Execute o instalador
4. IMPORTANTE: Marque ✅ "Add Python to PATH"
5. Clique em "Install Now"
6. Verificar no PowerShell:
```
```powershell
python --version    # Python 3.11.x
pip --version
```

---

## 5. Ambiente Virtual e Dependências Python

### 5.1 Criar Ambiente Virtual Isolado

**Ubuntu / macOS:**
```bash
cd ~/mba-rag
python3.11 -m venv venv_rag

# Ativar o ambiente
source venv_rag/bin/activate
# O prompt deve mostrar: (venv_rag)

# Verificar
which python   # ~/mba-rag/venv_rag/bin/python
```

**Windows (PowerShell):**
```powershell
Set-Location "$HOME\mba-rag"
python -m venv venv_rag

# Ativar (PowerShell)
.\venv_rag\Scripts\Activate.ps1

# Se der erro de política de execução:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verificar
Get-Command python   # deve apontar para venv_rag
```

### 5.2 Instalar Dependências do Curso

```bash
# Atualizar pip
pip install --upgrade pip

# Criar arquivo requirements.txt
cat > requirements.txt << 'EOF'
# ─── NLP e Embeddings ───────────────────────────────────────
sentence-transformers==3.0.1
transformers==4.44.0
tokenizers==0.19.1
FlagEmbedding==1.2.11        # BGE-M3

# ─── Busca Vetorial ─────────────────────────────────────────
faiss-cpu==1.8.0             # Use faiss-gpu se tiver GPU local
opensearch-py==2.7.1

# ─── LLM e Agentes ──────────────────────────────────────────
langchain==0.3.0
langchain-community==0.3.0
langchain-openai==0.2.0
openai>=1.40.0               # Compatível com vLLM

# ─── Observabilidade ────────────────────────────────────────
langfuse==2.36.1

# ─── Visualização ───────────────────────────────────────────
umap-learn==0.5.6
matplotlib==3.9.2
plotly==5.24.0
seaborn==0.13.2

# ─── Utilitários ────────────────────────────────────────────
pandas==2.2.2
numpy==1.26.4
tqdm==4.66.5
python-dotenv==1.0.1
requests==2.32.3
psutil==6.0.0
pyngrok>=7.0.0
jupyter==1.1.1
ipykernel==6.29.5
EOF

# Instalar todas as dependências
pip install -r requirements.txt

# Registrar kernel no Jupyter
python -m ipykernel install --user --name=venv_rag --display-name="MBA RAG (Python 3.11)"
```

**Saída esperada:**
```
Successfully installed sentence-transformers-3.0.1 ...
Installed kernelspec venv_rag in ~/.local/share/jupyter/kernels/venv_rag
```

### 5.3 Verificar Instalação

```bash
python -c "
import sentence_transformers, transformers, FlagEmbedding
import faiss, opensearchpy, langchain, langfuse, umap
print('✅ Todas as bibliotecas instaladas corretamente!')
print(f'   sentence-transformers: {sentence_transformers.__version__}')
print(f'   transformers:          {transformers.__version__}')
print(f'   langchain:             {langchain.__version__}')
"
```

---

## 6. vLLM — Servidor de Inferência LLM

O **vLLM** (Very Large Language Model serving) é o servidor de inferência de alto desempenho utilizado neste curso. Ele implementa **PagedAttention** para maximizar o uso de VRAM e serve modelos com API 100% compatível com o padrão OpenAI.

> **Por que vLLM?**  
> O vLLM é o padrão de mercado para deploy de LLMs em produção — incluindo ambientes de segurança pública e jurídico que rodam em Kubernetes (tema das aulas finais do curso). Aprender vLLM desde o início prepara você para a realidade dos sistemas RAG em produção. Este curso utiliza exclusivamente vLLM como servidor de inferência.

### 6.1 Pré-requisito: Token HuggingFace

O Llama 3.1 8B possui licença de uso restrito da Meta. É necessário aceitar a licença e obter um token:

```
1. Acesse: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Clique em "Agree and access repository" (aceitar a licença Meta Llama 3.1)
3. Acesse: https://huggingface.co/settings/tokens
4. Clique em "New token" → Tipo: "Read" → Nome: "mba-rag"
5. Copie o token (começa com hf_...)
6. Guarde — você precisará nos passos 6.2 e 8.1
```

### 6.2 Instalar vLLM (com GPU NVIDIA local)

> **⚠️ Requisito:** GPU NVIDIA com CUDA 12.x e pelo menos 16GB de VRAM para Llama 3.1 8B.  
> **Sem GPU local?** Pule para a seção 10 (Google Colab) — o Lab 3 usa a GPU T4 do Colab.

**Ubuntu (com GPU NVIDIA):**
```bash
# Passo 1: Verificar CUDA e GPU
nvidia-smi
nvcc --version    # CUDA deve ser 12.x

# Passo 2: Ativar ambiente virtual
source ~/mba-rag/venv_rag/bin/activate

# Passo 3: Instalar vLLM
# O vLLM requer PyTorch com suporte CUDA — instale na ordem abaixo
pip install torch==2.4.0+cu121 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

pip install vllm==0.5.5

# Passo 4: Verificar instalação
python -c "import vllm; print(f'✅ vLLM {vllm.__version__} instalado!')"
```

**macOS (Apple Silicon M1/M2/M3):**
```bash
# macOS com Apple Silicon usa MPS (Metal Performance Shaders)
# O vLLM tem suporte experimental para MPS
source ~/mba-rag/venv_rag/bin/activate

pip install vllm==0.5.5

# Verificar
python -c "import vllm; print(f'✅ vLLM {vllm.__version__}'); import torch; print(f'   MPS disponível: {torch.backends.mps.is_available()}')"
```

**Windows (PowerShell — com GPU NVIDIA):**
```powershell
# Ativar ambiente virtual
.\venv_rag\Scripts\Activate.ps1

# Instalar PyTorch com CUDA
pip install torch==2.4.0+cu121 torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu121

# Instalar vLLM
pip install vllm==0.5.5

# Verificar
python -c "import vllm; print(f'vLLM {vllm.__version__} instalado!')"
```

### 6.3 Iniciar o Servidor vLLM

**Ubuntu / macOS:**
```bash
# Configura o token HuggingFace
export HF_TOKEN="hf_seu_token_aqui"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Inicia o servidor vLLM em background
# Modelo: Llama 3.1 8B Instruct
# Porta: 8000 (padrão)
# GPU utilização: 90% da VRAM
# Contexto máximo: 4096 tokens (adequado para GPUs com 16GB)
nohup python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --trust-remote-code \
    --disable-log-requests \
    > /tmp/vllm_server.log 2>&1 &

echo "⏳ Servidor vLLM iniciando em background..."
echo "   Log: /tmp/vllm_server.log"
echo "   PID: $!"

# Aguardar carregamento do modelo (8-15 minutos na primeira vez)
echo "⏳ Aguardando modelo carregar (primeira vez: até 15 min)..."
for i in $(seq 1 90); do
    sleep 10
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
    if [ "$STATUS" = "200" ]; then
        echo "✅ Servidor vLLM pronto após $((i * 10))s!"
        break
    fi
    if [ $((i % 6)) -eq 0 ]; then
        echo "  [${i}0s] Ainda carregando..."
        tail -2 /tmp/vllm_server.log
    fi
done
```

**Windows (PowerShell):**
```powershell
# Configura token
$env:HF_TOKEN = "hf_seu_token_aqui"
$env:HUGGING_FACE_HUB_TOKEN = $env:HF_TOKEN

# Inicia vLLM em nova janela PowerShell
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    ".\venv_rag\Scripts\Activate.ps1; python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000 --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 4096 --trust-remote-code --disable-log-requests"
)

# Aguardar e verificar
Write-Host "⏳ Aguardando servidor vLLM iniciar..."
Start-Sleep -Seconds 60

# Verificar
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/v1/models" -TimeoutSec 10
    Write-Host "✅ Servidor vLLM ativo! Modelos: $($response.data.id -join ', ')"
} catch {
    Write-Host "⚠️ Servidor ainda carregando. Aguarde mais alguns minutos."
}
```

### 6.4 Verificar e Testar o Servidor vLLM

```bash
# Verificar saúde
curl -s http://localhost:8000/health
# Retorna: {} (status 200)

# Listar modelos disponíveis
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Teste de completion via curl
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "Você é um assistente jurídico objetivo."},
      {"role": "user", "content": "O que é peculato? Responda em 2 frases."}
    ],
    "temperature": 0.2,
    "max_tokens": 100
  }' | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('✅ Resposta:', d['choices'][0]['message']['content'])
print(f'   Tokens: {d[\"usage\"][\"total_tokens\"]}')
"
```

### 6.5 Expor vLLM via ngrok (para acesso externo)

Se precisar acessar o servidor vLLM do Colab (como faz com o OpenSearch):

```bash
# Ubuntu / macOS — Cria túnel para a porta 8000 do vLLM
ngrok http 8000 &

# Obtém a URL pública
sleep 3
curl -s http://localhost:4040/api/tunnels | python3 -c "
import json, sys
d = json.load(sys.stdin)
url = d['tunnels'][0]['public_url']
print(f'🌐 vLLM URL pública: {url}')
print(f'   Use em notebooks: {url}/v1/chat/completions')
"
```

```python
# Python: testar vLLM via ngrok
from openai import OpenAI

VLLM_NGROK_URL = "https://abc123.ngrok-free.app"  # Sua URL ngrok

client = OpenAI(
    base_url=f"{VLLM_NGROK_URL}/v1",
    api_key="vllm-local",
    default_headers={"ngrok-skip-browser-warning": "true"}
)

resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Olá! Diga apenas: OK"}],
    max_tokens=10
)
print("✅", resp.choices[0].message.content)
```

### 6.6 Iniciar vLLM Automaticamente no Boot (Ubuntu com systemd)

```bash
# Cria serviço systemd para vLLM
sudo tee /etc/systemd/system/vllm.service << 'EOF'
[Unit]
Description=vLLM Inference Server
After=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/mba-rag
Environment="HF_TOKEN=hf_seu_token_aqui"
Environment="HUGGING_FACE_HUB_TOKEN=hf_seu_token_aqui"
ExecStart=/home/YOUR_USER/mba-rag/venv_rag/bin/python \
    -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --trust-remote-code \
    --disable-log-requests
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Substitua YOUR_USER pelo seu usuário
sudo sed -i "s/YOUR_USER/$USER/g" /etc/systemd/system/vllm.service

# Habilitar e iniciar
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
sudo systemctl status vllm
```

### 6.7 Modelos Alternativos para Hardware Limitado

| Modelo | VRAM Necessária | Comando vLLM |
|--------|----------------|--------------|
| Llama 3.1 8B Instruct | 16 GB | `--model meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Llama 3.1 8B (AWQ 4-bit) | 8 GB | `--model casperhansen/llama-3-8b-instruct-awq --quantization awq` |
| Mistral 7B Instruct | 14 GB | `--model mistralai/Mistral-7B-Instruct-v0.3` |
| Phi-3 Mini Instruct | 8 GB | `--model microsoft/Phi-3-mini-4k-instruct` |

> **AWQ (Activation-aware Weight Quantization):** Reduz VRAM pela metade com perda mínima de qualidade. Ideal para GPUs com 8-10GB.

---

## 7. LangFuse

LangFuse é a plataforma de observabilidade para rastreamento de chamadas LLM e sistemas RAG.

### 7.1 Opção A — LangFuse Cloud (Recomendado para o Curso)

```
1. Acesse: https://cloud.langfuse.com
2. Crie uma conta gratuita (email ou GitHub)
3. Clique em "Create new project"
4. Nome: mba-rag-direito
5. Vá em Settings → API Keys → Create new API Key
6. COPIE E SALVE:
   - Public Key:  pk-lf-xxxxxxxxxxxxxxxx
   - Secret Key:  sk-lf-xxxxxxxxxxxxxxxx
7. Anote também o Host: https://cloud.langfuse.com
```

> **⚠️ Atenção:** A Secret Key só é exibida uma vez! Guarde em local seguro.

### 7.2 Opção B — LangFuse Self-Hosted (Para Dados Sigilosos)

Para ambientes com dados sensíveis (processos sigilosos, investigações em andamento):

```bash
# Passo 1: Clonar repositório LangFuse
git clone https://github.com/langfuse/langfuse.git
cd langfuse

# Passo 2: Configurações mínimas
cp .env.example .env
cat >> .env << 'EOF'
NEXTAUTH_SECRET=sua-chave-secreta-minimo-32-chars
SALT=seu-salt-aleatorio-para-hashing
ENCRYPTION_KEY=sua-chave-encriptacao-32-chars-hex
NEXTAUTH_URL=http://localhost:3000
LANGFUSE_INIT_ORG_NAME=MBA RAG
LANGFUSE_INIT_PROJECT_NAME=direito-seguranca
LANGFUSE_INIT_USER_EMAIL=admin@mba.local
LANGFUSE_INIT_USER_PASSWORD=Admin@1234!
EOF

# Passo 3: Iniciar
docker compose up -d
sleep 60
echo "LangFuse disponível em: http://localhost:3000"
```

### 7.3 Verificar Conexão LangFuse

```python
# Execute no terminal com venv_rag ativado
python3 -c "
from langfuse import Langfuse
import os

lf = Langfuse(
    public_key=os.environ.get('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.environ.get('LANGFUSE_SECRET_KEY'),
    host=os.environ.get('LANGFUSE_HOST', 'https://cloud.langfuse.com')
)

try:
    lf.auth_check()
    print('✅ LangFuse: conexão validada!')
except Exception as e:
    print(f'❌ Erro: {e}')
"
```

---

## 8. Variáveis de Ambiente

### 8.1 Criar Arquivo .env

```bash
# NUNCA faça commit deste arquivo! Adicione ao .gitignore
cat > ~/mba-rag/.env << 'EOF'
# ─── OpenSearch ──────────────────────────────────────────────
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_URL=http://localhost:9200

# ─── ngrok (preencha após criar o túnel) ─────────────────────
# Atualize estes valores sempre que recriar os túneis ngrok
OPENSEARCH_NGROK_URL=https://SEU_TUNEL_OPENSEARCH.ngrok-free.app
VLLM_NGROK_URL=https://SEU_TUNEL_VLLM.ngrok-free.app

# ─── vLLM ────────────────────────────────────────────────────
VLLM_HOST=localhost
VLLM_PORT=8000
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_DEFAULT=meta-llama/Meta-Llama-3.1-8B-Instruct

# ─── HuggingFace ─────────────────────────────────────────────
HF_TOKEN=hf_SEU_TOKEN_AQUI
HUGGING_FACE_HUB_TOKEN=hf_SEU_TOKEN_AQUI

# ─── LangFuse ────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY=pk-lf-SEU_TOKEN_AQUI
LANGFUSE_SECRET_KEY=sk-lf-SEU_TOKEN_AQUI
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PROJECT=mba-rag-direito

# ─── Configurações do Curso ──────────────────────────────────
EMBEDDING_MODEL_PRIMARY=BAAI/bge-m3
EMBEDDING_MODEL_SECONDARY=paraphrase-multilingual-mpnet-base-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=5
TEMPERATURE_DEFAULT=0.2
MAX_TOKENS_DEFAULT=512
EOF

# Proteger o arquivo
chmod 600 ~/mba-rag/.env

echo "✅ Arquivo .env criado!"
echo "⚠️  Edite o arquivo e preencha: HF_TOKEN, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY"
echo "⚠️  Atualize OPENSEARCH_NGROK_URL e VLLM_NGROK_URL após criar os túneis ngrok"
```

**Windows (PowerShell):**
```powershell
# Cria .env na pasta do projeto
$envContent = @"
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_NGROK_URL=https://SEU_TUNEL_OPENSEARCH.ngrok-free.app
VLLM_HOST=localhost
VLLM_PORT=8000
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_DEFAULT=meta-llama/Meta-Llama-3.1-8B-Instruct
HF_TOKEN=hf_SEU_TOKEN_AQUI
LANGFUSE_PUBLIC_KEY=pk-lf-SEU_TOKEN_AQUI
LANGFUSE_SECRET_KEY=sk-lf-SEU_TOKEN_AQUI
LANGFUSE_HOST=https://cloud.langfuse.com
"@

$envContent | Out-File -FilePath "$HOME\mba-rag\.env" -Encoding utf8
Write-Host "✅ .env criado em $HOME\mba-rag\.env"
```

### 8.2 Adicionar ao .gitignore

```bash
cat >> ~/mba-rag/.gitignore << 'EOF'
# Credenciais — NUNCA commitar!
.env
*.env
.env.*
!.env.example

# Dados e modelos grandes
datasets/
models/
*.bin
*.safetensors
*.gguf

# Python
__pycache__/
*.pyc
venv_rag/
EOF
```

### 8.3 Carregar Variáveis de Ambiente

```bash
# Ubuntu / macOS — Carrega para a sessão atual
export $(grep -v '^#' ~/mba-rag/.env | xargs)

# Verificar
echo $OPENSEARCH_URL          # http://localhost:9200
echo $VLLM_BASE_URL           # http://localhost:8000/v1
echo $HF_TOKEN                # hf_...

# Para carga automática em novas sessões, adicione ao ~/.bashrc:
echo 'export $(grep -v "^#" ~/mba-rag/.env | xargs) 2>/dev/null' >> ~/.bashrc
source ~/.bashrc
```

```powershell
# Windows PowerShell — Carrega variáveis do .env
Get-Content "$HOME\mba-rag\.env" | Where-Object { $_ -notmatch '^#' -and $_ -ne '' } | ForEach-Object {
    $key, $value = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')
}

Write-Host "✅ Variáveis de ambiente carregadas!"
```

---

## 9. Validação Final do Ambiente

Execute o script de validação abaixo para confirmar que tudo está funcionando:

```bash
# Salva o script de validação
cat > ~/mba-rag/validar_ambiente.py << 'PYEOF'
#!/usr/bin/env python3
"""
Script de Validação do Ambiente — MBA RAG & CAG
Aula 1: OpenSearch + vLLM + LangFuse + ngrok
"""
import sys, os, time, requests
from datetime import datetime

resultados = []
avisos = []

def teste(nome, fn, opcional=False):
    try:
        fn()
        resultados.append(('✅', nome, ''))
    except Exception as e:
        if opcional:
            resultados.append(('⚠️ ', f'{nome} (opcional)', str(e)[:80]))
        else:
            resultados.append(('❌', nome, str(e)[:80]))

print('=' * 65)
print(f'VALIDAÇÃO DO AMBIENTE MBA RAG — {datetime.now().strftime("%d/%m/%Y %H:%M")}')
print('=' * 65)

# 1. Python 3.11+
def check_python():
    assert sys.version_info >= (3, 11), f"Python 3.11+ necessário, encontrado: {sys.version}"
teste("Python 3.11+", check_python)

# 2. Bibliotecas NLP
def check_nlp():
    import sentence_transformers, transformers, FlagEmbedding
teste("sentence-transformers + BGE-M3", check_nlp)

# 3. FAISS
def check_faiss():
    import faiss
    idx = faiss.IndexFlatL2(128)
    assert idx.d == 128
teste("FAISS (busca vetorial)", check_faiss)

# 4. opensearch-py
def check_opensearch_lib():
    from opensearchpy import OpenSearch
teste("opensearch-py (biblioteca)", check_opensearch_lib)

# 5. OpenSearch servidor
def check_opensearch_server():
    r = requests.get("http://localhost:9200", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert "version" in body, "Resposta inesperada do servidor"
teste("OpenSearch 3.x (servidor local)", check_opensearch_server)

# 6. LangChain
def check_langchain():
    import langchain
    assert langchain.__version__.startswith("0.3"), f"Versão incorreta: {langchain.__version__}"
teste("LangChain 0.3.x", check_langchain)

# 7. vLLM servidor
def check_vllm_server():
    r = requests.get("http://localhost:8000/health", timeout=5)
    assert r.status_code == 200, f"vLLM respondeu HTTP {r.status_code}"
    r2 = requests.get("http://localhost:8000/v1/models", timeout=5)
    modelos = [m['id'] for m in r2.json().get('data', [])]
    assert len(modelos) > 0, "Nenhum modelo carregado no vLLM"
    print(f"\n      Modelo: {modelos[0]}", end="")
teste("vLLM (servidor + modelo carregado)", check_vllm_server)

# 8. vLLM geração de texto
def check_vllm_completion():
    from openai import OpenAI
    r = requests.get("http://localhost:8000/v1/models", timeout=5)
    model_id = r.json()['data'][0]['id']
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Responda: OK"}],
        max_tokens=5,
        temperature=0.0
    )
    assert len(resp.choices[0].message.content) > 0
teste("vLLM geração de texto (API /v1/chat/completions)", check_vllm_completion)

# 9. LangFuse
def check_langfuse():
    from langfuse import Langfuse
    pk = os.environ.get('LANGFUSE_PUBLIC_KEY', '')
    sk = os.environ.get('LANGFUSE_SECRET_KEY', '')
    assert pk and not pk.endswith('_AQUI'), "Configure LANGFUSE_PUBLIC_KEY no .env"
    lf = Langfuse(public_key=pk, secret_key=sk,
                  host=os.environ.get('LANGFUSE_HOST', 'https://cloud.langfuse.com'))
    lf.auth_check()
teste("LangFuse (observabilidade)", check_langfuse)

# 10. ngrok (opcional)
def check_ngrok():
    r = requests.get("http://localhost:4040/api/tunnels", timeout=3)
    tunnels = r.json().get('tunnels', [])
    assert len(tunnels) > 0, "Nenhum túnel ngrok ativo"
    urls = [t['public_url'] for t in tunnels]
    print(f"\n      Túneis: {urls}", end="")
teste("ngrok (túnel ativo)", check_ngrok, opcional=True)

# 11. GPU CUDA
def check_gpu():
    import torch
    assert torch.cuda.is_available(), "GPU CUDA não detectada"
    gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    name = torch.cuda.get_device_properties(0).name
    print(f"\n      {name} ({gb}GB)", end="")
    assert gb >= 4, f"GPU com apenas {gb}GB VRAM — mínimo 4GB"
teste("GPU CUDA (4GB+ VRAM)", check_gpu, opcional=True)

# 12. Bibliotecas de visualização
def check_viz():
    import umap, matplotlib, seaborn, plotly
teste("Bibliotecas de visualização (UMAP, plotly, seaborn)", check_viz)

# ── Resultados ────────────────────────────────────────────────
print()
for status, nome, detalhe in resultados:
    print(f"  {status}  {nome}")
    if detalhe and status == '❌':
        print(f"       → {detalhe}")

obrigatorios_ok = sum(1 for s, _, _ in resultados if s == '✅')
opcionais_ok    = sum(1 for s, _, _ in resultados if s == '⚠️ ')
total           = len(resultados)

print()
print(f"Obrigatórios: {obrigatorios_ok}/{total - 2} ✅ | Opcionais: {opcionais_ok}")

falhas = [n for s, n, _ in resultados if s == '❌']
if not falhas:
    print()
    print("🎉 AMBIENTE COMPLETAMENTE VALIDADO!")
    print("   Você está pronto para os laboratórios da Aula 1.")
else:
    print()
    print("⚠️  Falhas detectadas:")
    for f in falhas:
        print(f"   ❌ {f}")
    print("   Consulte a seção de troubleshooting deste roteiro.")
PYEOF

# Executa a validação
cd ~/mba-rag
source venv_rag/bin/activate
export $(grep -v '^#' .env | xargs) 2>/dev/null
python3 validar_ambiente.py
```

**Saída esperada (100% OK):**
```
=================================================================
VALIDAÇÃO DO AMBIENTE MBA RAG — 14/04/2026 14:30
=================================================================

  ✅  Python 3.11+
  ✅  sentence-transformers + BGE-M3
  ✅  FAISS (busca vetorial)
  ✅  opensearch-py (biblioteca)
  ✅  OpenSearch 3.x (servidor local)
  ✅  LangChain 0.3.x
  ✅  vLLM (servidor + modelo carregado)
        Modelo: meta-llama/Meta-Llama-3.1-8B-Instruct
  ✅  vLLM geração de texto (API /v1/chat/completions)
  ✅  LangFuse (observabilidade)
  ⚠️   ngrok (túnel ativo) (opcional)
        Túneis: ['https://abc123.ngrok-free.app']
  ⚠️   GPU CUDA (4GB+ VRAM) (opcional)
        NVIDIA GeForce RTX 3080 (10GB)
  ✅  Bibliotecas de visualização (UMAP, plotly, seaborn)

Obrigatórios: 10/10 ✅ | Opcionais: 2

🎉 AMBIENTE COMPLETAMENTE VALIDADO!
   Você está pronto para os laboratórios da Aula 1.
```

---

## 10. Configuração Google Colab (Alternativa Cloud)

Para alunos sem GPU local ou que preferem ambiente cloud. Nesta configuração:
- **vLLM + Llama 3.1 8B** rodam no Colab (GPU T4 gratuita)
- **OpenSearch** roda local (Docker Desktop) e é acessado via ngrok

### 10.1 Configurar Segredos no Colab

```
1. Abra o Google Colab: https://colab.research.google.com
2. Clique em 🔑 "Segredos" no painel esquerdo
3. Adicione os seguintes segredos (ative o toggle "Acesso ao notebook" em cada um):

   ┌────────────────────────────────┬─────────────────────────────────────────────────┐
   │ Nome                           │ Valor                                           │
   ├────────────────────────────────┼─────────────────────────────────────────────────┤
   │ HF_TOKEN                       │ hf_seu_token_huggingface_aqui                   │
   │ OPENSEARCH_NGROK_URL           │ https://abc123.ngrok-free.app (URL ngrok local) │
   │ LANGFUSE_PUBLIC_KEY            │ pk-lf-seu-token-aqui                            │
   │ LANGFUSE_SECRET_KEY            │ sk-lf-seu-token-aqui                            │
   │ LANGFUSE_HOST                  │ https://cloud.langfuse.com                      │
   │ NGROK_AUTHTOKEN                │ seu_token_ngrok_aqui (para expor vLLM)          │
   └────────────────────────────────┴─────────────────────────────────────────────────┘
```

### 10.2 Configurar Runtime GPU T4

```
1. Menu → Runtime → Alterar tipo de runtime
2. Acelerador de hardware: GPU
3. Tipo de GPU: T4 (gratuito) ou A100 (Colab Pro)
4. Salvar
```

### 10.3 Instalar Dependências no Colab

```python
# Cole no início de cada notebook do curso:
# Esta célula instala todas as dependências necessárias no Colab

DEPENDENCIAS = [
    "sentence-transformers==3.0.1",
    "FlagEmbedding==1.2.11",
    "faiss-cpu==1.8.0",
    "opensearch-py==2.7.1",
    "langchain==0.3.0",
    "langchain-community==0.3.0",
    "langchain-openai==0.2.0",
    "openai>=1.40.0",
    "langfuse==2.36.1",
    "umap-learn==0.5.6",
    "pyngrok>=7.0.0",
    "vllm==0.5.5",       # Somente nos labs que usam vLLM
]

import subprocess
for dep in DEPENDENCIAS:
    subprocess.run(["pip", "install", dep, "--quiet"], check=True)

print("✅ Dependências instaladas!")
```

---

## 🔧 TROUBLESHOOTING GERAL

| Problema | Causa Mais Comum | Solução |
|----------|-----------------|---------|
| OpenSearch não inicia (Ubuntu) | `vm.max_map_count` baixo | `sudo sysctl -w vm.max_map_count=262144` |
| OpenSearch não inicia (Windows/macOS) | Memória insuficiente para Docker Desktop | Aumente a memória em Docker Desktop → Settings → Resources |
| `Permission denied` no Docker (Ubuntu) | Usuário fora do grupo docker | `sudo usermod -aG docker $USER && newgrp docker` |
| vLLM `CUDA out of memory` | VRAM insuficiente para o modelo | Use modelo AWQ quantizado ou reduza `--max-model-len` |
| vLLM demora >15min para iniciar | Download do modelo na primeira vez | Aguarde; verifique `tail -f /tmp/vllm_server.log` |
| `401 Unauthorized` no HuggingFace | Token inválido ou licença não aceita | Aceite a licença em huggingface.co/meta-llama e gere novo token |
| ngrok `Your connection to ngrok...` no browser | Página de aviso do plano gratuito | Adicione header `ngrok-skip-browser-warning: true` nas requisições |
| ngrok URL muda a cada reinício | Comportamento do plano gratuito | Use plano pago para URL fixa ou salve a URL no .env manualmente |
| `ImportError` nas bibliotecas | Ambiente virtual não ativado | `source ~/mba-rag/venv_rag/bin/activate` |
| LangFuse `401 Unauthorized` | Chaves inválidas ou expiradas | Regenere as chaves no dashboard LangFuse |
| OpenSearch status RED | Problema de inicialização | `docker logs opensearch-node1 \| tail -50` |
| `docker compose` não encontrado | Docker Compose V2 não instalado | `sudo apt-get install docker-compose-plugin` |
| Windows: `Activate.ps1 não pode ser carregado` | Política de execução restrita | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## 📁 ESTRUTURA DE DIRETÓRIOS RECOMENDADA

```
~/mba-rag/
├── .env                     # Credenciais (NUNCA no git!)
├── .gitignore
├── requirements.txt
├── validar_ambiente.py
├── infra/
│   └── docker-compose.yml   # OpenSearch + Dashboards
├── venv_rag/                # Ambiente virtual Python 3.11
├── aula1/
│   ├── teoria/
│   │   └── AULA1_TEORIA.md
│   ├── labs/
│   │   ├── LAB1_OpenSearch_Docker.ipynb
│   │   ├── LAB2_Setup_Colab_Ambiente.ipynb
│   │   ├── LAB3_vLLM_LLM_Local.ipynb
│   │   ├── LAB4_LangFuse_Observabilidade.ipynb
│   │   └── LAB5_Embeddings_BGE_M3_UMAP.ipynb
│   ├── exemplos/
│   │   └── EXEMPLO_Pipeline_RAG_Minimo.ipynb
│   └── datasets/
│       └── corpus_juridico_aula1.json
└── aula2/
    └── ...
```

---

*Roteiro de Instalação v2.0 — MBA RAG & CAG Aplicados a Direito e Segurança Pública*  
*Conforme boas práticas de segurança da informação para ambientes jurídicos (LGPD, sigilo funcional)*
