# Projeto CDIA Brando

Repositório para documentação e código do Projeto I e II em CDIA relacionados ao BRANDO.

## ToC

1. [Authors](#authors)
1. [Reports](#reports)
1. [Installation and Usage](#installation-and-usage)

## Authors

- Leonardo Mallmann
- Arthur Bianchessi
- Arthur Germano
- Bento Bastian
- Carlos Gomes

[top](#toc)

## Reports

- [Relatório de Andamento][relatorio-andamento]
- [Relatório Final][relatorio-final]

[relatorio-andamento]:https://docs.google.com/document/d/1wNaZKtH8gjUxu48lsEJ2tiqXKHV6XwN5XUMkmQ-0_tc/edit?usp=sharing
[relatorio-final]:https://docs.google.com/document/d/1jmKnT9WTnNtdhsgM5pYGcB1qTTl676wySKUbuXro6u8/edit?usp=sharing

[top](#toc)

## Installation and Usage

### Hugging Face

1. Create a hugging face account.

1. Go to profile -> settings -> access tokens -> create new token (write type).

1. Copy the token and paste it in the `.env` file in the root of the project with the name `HUGGINGFACEHUB_API_TOKEN`:

    ```txt
    HUGGINGFACEHUB_API_TOKEN=YOUR_TOKEN
    ```

### Github

1. Clone this project on your local machine:

    ```sh
    git clone git@github.com:mallmann02/projeto_cdia_brando.git
    ```

### Virtual environment and dependencies

1. Change your working directory to the project's root dir:

    ```sh
    cd projeto_cdia_brando
    ```

1. Create a virtual environment and activate it:

    ```sh
    python3 -m venv .venv/
    source .venv/bin/activate
    ```

1. Install the dependecies:

    ```sh
    pip install -r requirements.txt
    ```

### Dataset

1. Download the `zip` data files [from Moodle](https://moodle.pucrs.br/course/view.php?id=84076) and place them in the `pac_data/` subdir.

### Running

1. Run the `main.ipynb` notebook.

1. When done, deactivate the virtual environment:

    ```sh
    deactivate
    ```

1. (Optional.) If you're completely done with this project, delete `.venv/` to save space.

[top](#toc)
