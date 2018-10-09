import re
from typing import Dict

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer

from modules.data.dataset_readers.copynet import CopyNetDatasetReader
from modules.data.tokenizers.word_splitter import NL2BashWordSplitter


utilities = [  # pylint: disable=invalid-name
        "find",
        "xargs",
        "grep",
        "rm",
        "ls",
        "echo",
        "sort",
        "chmod",
        "wc",
        "cut",
        "head",
        "chown",
        "cat",
        "mv",
        "cp",
        "mkdir",
        "tail",
        "dirname",
        "tr",
        "uniq",
        "split",
        "tar",
        "readlink",
        "tee",
        "basename",
        "ln",
        "read",
        "rsync",
        "which",
        "mount",
        "ssh",
        "file",
        "pwd",
        "du",
        "md5sum",
        "ifconfig",
        "shopt",
        "od",
        "cd",
        "comm",
        "diff",
        "hostname",
        "df",
        "rename",
        "mktemp",
        "date",
        "nl",
        "column",
        "dig",
        "paste",
        "history",
        "rev",
        "zcat",
        "touch",
        "cal",
        "chgrp",
        "whoami",
        "ping",
        "gzip",
        "rmdir",
        "seq",
        "tree",
        "tac",
        "bzip2",
        "fold",
        "join",
        "cpio",
        "who",
        "pstree",
        "uname",
        "env",
        "kill",
]


@DatasetReader.register("nl2bash")
class NL2BashDatasetReader(CopyNetDatasetReader):

    prefix_finder = re.compile(r"(\s|^)(/bin/|/usr/bin/)(" + r"|".join(utilities) + r")(\s|$)")
    sudo_finder = re.compile(r"^sudo\s")
    prompt_finder = re.compile(r"^(\$|#)\s?")

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        source_tokenizer = source_tokenizer or WordTokenizer(word_splitter=NL2BashWordSplitter())
        target_tokenizer = target_tokenizer or source_tokenizer
        super().__init__(source_tokenizer=source_tokenizer,
                         target_tokenizer=target_tokenizer,
                         source_token_indexers=source_token_indexers,
                         target_token_indexers=target_token_indexers,
                         lazy=lazy)

    @overrides
    def _preprocess_target(self, target_string: str) -> str:
        target_string = self.prefix_finder.sub(r"\g<1>\g<3>\g<4>", target_string)
        target_string = self.sudo_finder.sub("", target_string)
        target_string = self.prompt_finder.sub("", target_string)
        return target_string
