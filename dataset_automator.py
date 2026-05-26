from google import genai
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()
CHAVE_API = os.getenv("GEMINI_API_KEY")

if not CHAVE_API:
    raise ValueError("Chave da API não encontrada! Verifique o arquivo .env.")

client = genai.Client(api_key=CHAVE_API)

topicos = [
    "Estruturas de Repetição (Loops)",
    "Manipulação de Strings e Caracteres",
    "Vetores / Arrays Unidimensionais",
    "Matrizes (Arrays Multidimensionais)",
    "Funções e Passagem de Parâmetros",
    "Recursão",
    "Algoritmos de Busca",
    "Algoritmos de Ordenação (Sorting)",
    "Ponteiros, Referências e Alocação Dinâmica",
    "Estruturas de Dados Compostas (Structs / Registros)",
    "Listas Encadeadas (Simples, Duplas ou Circulares)",
    "Pilhas e Filas (Stacks & Queues)",
    "Tabelas Hash (Mapas / Dicionários)",
    "Árvores Binárias (e Árvores Binárias de Busca - BST)",
    "Grafos Básicos",
    "Manipulação de Arquivos",
    "Programação Orientada a Objetos (POO)",
    "Tratamento de Exceções",
    "Entrada/Saída Padrão e Tipos de Dados",
    "Estruturas Condicionais",
]

prompt_base = """Você é um engenheiro de dados construindo um dataset de treinamento para um modelo de Inteligência Artificial.
O objetivo do modelo que será treinado é gerar exercícios de programação para um Autograder.
Sua tarefa é gerar 5 exemplos de treinamento INÉDITOS sobre o tópico: {topico}.
Para CADA exemplo, você deve simular um usuário real pedindo o exercício (com variações de vocabulário, tamanho da frase e tom) e, em seguida, gerar a saída perfeita.
Dificuldade: Misture 2 fáceis, 2 médios e 1 difícil.

REGRAS ESTRITAS DE ESTRUTURA E CONTEÚDO:
1. Você DEVE retornar EXATAMENTE 5 linhas de texto. Nenhuma palavra a mais.
2. Cada JSON deve ter EXATAMENTE duas chaves principais: "instrucao_simulada" e "saida_esperada".
3. CADA linha deve ser um objeto JSON válido, independente, no formato JSONL.
4. O valor de "instrucao_simulada" deve variar bastante. Exemplo: um professor pedindo uma questão de prova para seus alunos, um treinador de maratona de programação querendo um desafio difícil, ou um pedido curto e direto.
5. NÃO use blocos de código Markdown (não escreva ```json).
6. O valor de "saida_esperada" deve ser um OBJETO JSON contendo a estrutura estrita do Autograder: title, description, input_spec, output_spec, constraints, examples e test_cases.
7. O campo 'input_spec' e 'output_spec' devem explicar claramente como os dados entram e saem pelo terminal (stdin/stdout).
8. O campo 'constraints' DEVE ser um array de strings com os limites das variáveis.
9. Os campos 'examples' e 'test_cases' DEVEM ser arrays de objetos, contendo EXATAMENTE as chaves "input" e "output". Devem haver PELO MENOS 3 exemplos e 3 test cases, e eles DEVEM ser diferentes entre si.
10. Os valores de "input" e "output" DEVEM OBRIGATORIAMENTE ser strings, e as quebras de linha reais do terminal devem ser escapadas como '\\n'.

EXEMPLO DE ESTRUTURA DA LINHA ESPERADA:
{{"instrucao": "Gere um exercício de programação sobre Estruturas Condicionais para uma prova.", "saida_esperada": "{{\\"title\\": \\"Classificação de Triângulos\\", \\"description\\": \\"Dados três valores inteiros representando os lados de um triângulo, determine se eles formam um triângulo válido. Se formarem, classifique-o em Equilátero (três lados iguais), Isósceles (dois lados iguais) ou Escaleno (todos os lados diferentes).\\", \\"input_spec\\": \\"A entrada contém uma única linha com três inteiros A, B e C (1 <= A, B, C <= 10^4) separados por um espaço.\\", \\"output_spec\\": \\"Imprima 'INVALIDO' se os lados não formarem um triângulo. Caso contrário, imprima 'EQUILATERO', 'ISOSCELES' ou 'ESCALENO' em letras maiúsculas.\\", \\"constraints\\": [\\"1 <= A, B, C <= 10^4\\", \\"A condição de existência diz que um lado deve ser menor que a soma dos outros dois.\\"], \\"examples\\": [{{\\"input\\": \\"3 3 3\\", \\"output\\": \\"EQUILATERO\\"}}, {{\\"input\\": \\"3 4 5\\", \\"output\\": \\"ESCALENO\\"}}], \\"test_cases\\": [{{\\"input\\": \\"3 3 3\\", \\"output\\": \\"EQUILATERO\\"}}, {{\\"input\\": \\"3 4 5\\", \\"output\\": \\"ESCALENO\\"}}, {{\\"input\\": \\"4 4 2\\", \\"output\\": \\"ISOSCELES\\"}}, {{\\"input\\": \\"1 2 5\\", \\"output\\": \\"INVALIDO\\"}}, {{\\"input\\": \\"10 5 5\\", \\"output\\": \\"INVALIDO\\"}}]}}"}}
"""

lotes_por_topico = 3

with open("meu_dataset.jsonl", "a", encoding="utf-8") as arquivo:
    for topico in topicos:
        print(f"Gerando exercícios para: {topico}...")

        for lote in range(lotes_por_topico):
            print(f"Gerando lote {lote + 1} de {lotes_por_topico}...")
        
            tentativa = 1
            while(1):
                try:
                    print("Enviando requisição para a API...")
                    resposta = client.models.generate_content(
                        model='gemma-4-31b-it',
                        contents=prompt_base.format(topico=topico),
                        config=genai.types.GenerateContentConfig(
                            temperature=0.8, 
                        )
                    )
                    print("Resposta gerada")
                    linhas_geradas = resposta.text.strip().split('\n')
                    
                    for linha in linhas_geradas:
                        try:
                            json_valido = json.loads(linha.strip())
                            instrucao_real = json_valido["instrucao_simulada"]
                            conteudo_saida = json_valido["saida_esperada"]
                            if isinstance(conteudo_saida, str):
                                conteudo_saida = json.loads(conteudo_saida)

                            saida_real = json.dumps(conteudo_saida, ensure_ascii=False)

                            linha_final = {
                                "instrucao": instrucao_real,
                                "saida_esperada": saida_real
                            }
                            
                            arquivo.write(json.dumps(linha_final, ensure_ascii=False) + '\n')
                        except json.JSONDecodeError:
                            print("Modelo gerou uma linha que não era JSON perfeito. Descartando...")

                    break

                except Exception as e:
                    erro_str = str(e)
                    if "503" in erro_str or "429" in erro_str:
                        print(f"Erro: {erro_str}. Aguardando 10s para tentar de novo (Tentativa {tentativa})...")
                        tentativa += 1
                        time.sleep(10)
                    else:
                        print(f"Erro fatal inesperado: {erro_str}")
                        break

            time.sleep(4)

print("Produção de dados concluída!")