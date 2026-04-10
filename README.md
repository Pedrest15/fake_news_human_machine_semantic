# noticias_falsas_humano_maquina_semantica

## Extração de regras ED via eud-portugues

O diretório `ed_rules/` contém o pipeline de extração de padrões de dependência
sobre arquivos CoNLL-U. Para enriquecer os arquivos com Enhanced UD antes da
extração, integramos [alvelvis/eud-portugues](https://github.com/alvelvis/eud-portugues)
As regras Grew (`conjunto_regras_porttinari.grs`) estão em `ed_rules/`.

### Setup

```bash
# Instale o Grew: https://grew.fr/usage/install/
uv sync
```

### Catalogar as regras Grew em JSON

```bash
python3 -m ed_rules.grs_catalog \
    ed_rules/conjunto_regras_porttinari.grs \
    aggregated_output/eud_rules_catalog.json
```

### Aplicar EUD a um diretório de .conllu

```bash
python3 -m ed_rules.eud_runner \
    portparser_results/fake_true_human \
    portparser_results_eud/fake_true_human \
    --strategy eud_portuguese
```

### Pipeline completo

O `main.py` sempre pré-processa os `.conllu` com as regras Grew do
eud-portugues antes da extração — não há flag para desativar.

```bash
python3 ed_rules/main.py
```