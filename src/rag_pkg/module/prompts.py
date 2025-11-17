from typing import Any, Dict

import yaml
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate


def build_qa_prompt(
    system_message: str,
    add_context: bool = True,
    add_history: bool = True,
    examples: Dict = None,
) -> PromptTemplate:
    qa_template = ""

    if add_context:
        qa_template += "Context: {context}\n"

    if add_history:
        qa_template += "History: {chat_history}\n"

    qa_template += "Question: {question}"

    full_template = system_message + "\n\n" + qa_template

    if examples:
        example_template = PromptTemplate(
            template="Question: {question}\nAnswer: {answer}",
            input_variables=["question", "answer"],
        )
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix=system_message + "\n\n",
            suffix=qa_template,
            input_variables=["context", "chat_instory", "question"],
            example_separator="\n\n",
        )

    else:
        prompt = PromptTemplate(
            template=full_template,
            input_variables=["context", "chat_instory", "question"],
        )

    return prompt


def save_prompt(prompt: PromptTemplate, save_path: str) -> None:

    prompt_data = {
        "_type": "prompt",
        "input_variables": prompt.input_variables,
        "metadata": prompt.metadata,
        "name": prompt.name,
        "optional_variables": prompt.optional_variables,
        "output_parser": prompt.output_parser,
        "partial_variables": prompt.partial_variables,
        "tags": prompt.tags,
        "template": prompt.template,
        "template_format": prompt.template_format,
        "validate_template": prompt.validate_template,
    }

    with open(save_path, "w", encoding="utf-8") as file:
        yaml.dump(prompt_data, file, allow_unicode=True)


def save_fewshot_prompt(prompt: Any, save_path: str) -> None:

    if hasattr(prompt.example_prompt, "__dict__"):
        example_prompt_dict = prompt.example_prompt.__dict__
    else:
        example_prompt_dict = prompt.example_prompt

    prompt_data = {
        "_type": "few_shot",
        "example_prompt": {
            "_type": "prompt",
            "input_variables": example_prompt_dict.get("input_variables", []),
            "metadata": example_prompt_dict.get("metadata", None),
            "name": example_prompt_dict.get("name", None),
            "optional_variables": example_prompt_dict.get("optional_variables", []),
            "output_parser": example_prompt_dict.get("output_parser", None),
            "partial_variables": example_prompt_dict.get("partial_variables", {}),
            "tags": example_prompt_dict.get("tags", None),
            "template": example_prompt_dict.get("template", ""),
            "template_format": example_prompt_dict.get("template_format", "f-string"),
            "validate_template": example_prompt_dict.get("validate_template", False),
        },
        "example_selector": prompt.example_selector,
        "example_separator": prompt.example_separator,
        "examples": prompt.examples,
        "input_variables": prompt.input_variables,
        "metadata": prompt.metadata,
        "name": prompt.name,
        "optional_variables": prompt.optional_variables,
        "output_parser": prompt.output_parser,
        "partial_variables": prompt.partial_variables,
        "prefix": prompt.prefix,
        "suffix": prompt.suffix,
        "tags": prompt.tags,
        "template_format": prompt.template_format,
        "validate_template": prompt.validate_template,
    }

    with open(save_path, "w", encoding="utf-8") as file:
        yaml.dump(prompt_data, file, allow_unicode=True)
