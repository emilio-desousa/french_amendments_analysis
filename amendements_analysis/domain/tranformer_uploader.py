from git import Repo, RemoteProgress
import git.cmd as git
import amendements_analysis.settings.base as stg
import os
import shutil
import os.path
from git import *
import git, os, shutil


class MyProgressPrinter(RemoteProgress):
    """Class to generate prints from git commands

    Parameters
    ----------
    RemoteProgress : gitpython Object
        progress from git
    """

    def new_message_handler(self):
        def handler(line):
            print(line.rstrip())
            return self._parse_progress_line(line.rstrip())

        return handler


class Model_Publisher:
    """Publish the trained tranformer model from tmp_model dir"""

    def __init__(self):
        """Clone the tranformer Camembert_aux_amandes and create the Repo object from gitPython"""
        if not os.path.exists(stg.CUSTOM_MODEL_REPO_DIR):
            self.repo = Repo.clone_from(
                stg.MODEL_REPO_URL,
                stg.CUSTOM_MODEL_REPO_DIR,
                progress=MyProgressPrinter(),
            )
        else:
            self.repo = Repo(stg.CUSTOM_MODEL_REPO_DIR)
        # self.repo.git.execute(["git", "lfs", "install"])
        self.repo.git.checkout("model1")

    def push_model(self):
        """Push the trained model in the remote git repository
        Add lfs support to deal with pytorch_model.bin because of his size
        """
        file_names = os.listdir(stg.TMP_MODEL_DIR)
        for file in file_names:
            shutil.move(
                os.path.join(stg.TMP_MODEL_DIR, file),
                os.path.join(stg.CUSTOM_MODEL_REPO_DIR, file),
            )

        self._install_lfs()
        self.repo.git.add(all=True)
        self.repo.index.commit("Auto push from package")
        origin = self.repo.remote()
        self.repo.create_head("test")
        origin.push("test", progress=MyProgressPrinter())

    def _install_lfs(self):
        self.repo.git.execute(["git", "lfs", "install"])
        self.repo.git.execute(
            ["git", "lfs", "migrate", "import", "--include='*.bin'", "--everything"]
        )
        self.repo.git.execute(["git", "lfs", "fetch"])
        self.repo.git.execute(["git", "lfs", "track", "pytorch_model.bin"])


# repo = Repo(stg.MODEL_DIR)
