"""Cross-profile cron delivery must not lend the source profile's bindings."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cron import scheduler_delivery
from hermes_cli import env_loader


class BotChatProfileEnvTests(unittest.TestCase):
    def test_destination_does_not_inherit_source_only_bindings(self):
        with tempfile.TemporaryDirectory(prefix="cron-profile-env-test-") as tmp:
            home = Path(tmp)
            root = home / ".hermes"
            source = root / "profiles" / "source"
            source.mkdir(parents=True)
            (source / ".env").write_text(
                "GOG_HOME=/synthetic/source/gog\n"
                "GOG_KEYRING_PASSWORD=synthetic-source-only\n"
                "SHARED_TEST_KEY=synthetic-source\n",
                encoding="utf-8",
            )
            (root / ".env").write_text(
                "SHARED_TEST_KEY=synthetic-target\n", encoding="utf-8"
            )
            for directory in (root, source):
                (directory / ".env").chmod(0o600)

            observed = {}

            def destination_startup(argv, **kwargs):
                observed["argv"] = argv
                # The CLI's -p default resolves its home before dotenv imports.
                with patch.dict(os.environ, kwargs["env"], clear=True):
                    os.environ["HERMES_HOME"] = str(root)
                    env_loader.load_hermes_dotenv(
                        hermes_home=root, load_external_secrets=False
                    )
                    observed["env"] = dict(os.environ)
                return subprocess.CompletedProcess(argv, 0, "", "")

            with (
                patch.object(Path, "home", return_value=home),
                patch.dict(
                    os.environ,
                    {
                        "HOME": str(home),
                        "HERMES_HOME": str(source),
                        "PATH": "/synthetic/bin:/usr/bin",
                    },
                    clear=True,
                ),
                patch.object(env_loader, "_apply_managed_env"),
                patch.object(env_loader, "_reapply_terminal_config_bridge"),
                patch.object(scheduler_delivery.shutil, "which", return_value="/synthetic/bin/hermes"),
                patch.object(scheduler_delivery, "_get_bot_chat_delivery_timeout", return_value=5),
                patch.object(scheduler_delivery.subprocess, "run", side_effect=destination_startup),
            ):
                env_loader.load_hermes_dotenv(
                    hermes_home=source, load_external_secrets=False
                )
                error = scheduler_delivery._deliver_to_bot_chat(
                    {"id": "synthetic-job", "name": "Synthetic cache failure"},
                    "Synthetic read failure; no real sources accessed.",
                    "default",
                )

            self.assertIsNone(error)
            self.assertEqual(observed["argv"][1:3], ["-p", "default"])
            child = observed["env"]
            self.assertEqual(child["HOME"], str(home))
            self.assertEqual(child["PATH"], "/synthetic/bin:/usr/bin")
            self.assertEqual(child["SHARED_TEST_KEY"], "synthetic-target")
            self.assertNotIn("GOG_HOME", child)
            self.assertNotIn("GOG_KEYRING_PASSWORD", child)


    def _captured_child_env(self, *, profile, multiplex=False, scope=None):
        from agent import secret_scope
        from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
        observed = {}
        with tempfile.TemporaryDirectory(prefix="cron-env-identity-") as tmp:
            home = Path(tmp)
            source = home / ".hermes/profiles/source"
            source.mkdir(parents=True)
            process_env = {
                "HOME": str(home), "HERMES_HOME": str(source), "PATH": "/usr/bin",
                "VENDOR_PROJECT": "source-project", "VENDOR_API_KEY": "synthetic-source",
                "SystemRoot": "synthetic-windows-root", "APPDATA": "synthetic-user-appdata",
                DELEGATED_CHILD_ENV_MARKER: "1",
            }
            def spawn(argv, **kwargs):
                observed.update(kwargs["env"])
                return subprocess.CompletedProcess(argv, 0, "", "")
            previous = secret_scope.is_multiplex_active()
            token = secret_scope.set_secret_scope(scope)
            try:
                secret_scope.set_multiplex_active(multiplex)
                with (
                    patch.object(Path, "home", return_value=home),
                    patch.dict(os.environ, process_env, clear=True),
                    patch.object(scheduler_delivery.shutil, "which", return_value="/usr/bin/hermes"),
                    patch.object(scheduler_delivery.subprocess, "run", side_effect=spawn),
                    patch.object(scheduler_delivery, "_get_bot_chat_delivery_timeout", return_value=5),
                ):
                    before = dict(os.environ)
                    self.assertIsNone(scheduler_delivery._deliver_to_bot_chat(
                        {"id": "synthetic-job"}, "Synthetic notification", profile))
                    self.assertEqual(dict(os.environ), before)
            finally:
                secret_scope.reset_secret_scope(token)
                secret_scope.set_multiplex_active(previous)
        self.assertEqual(observed[DELEGATED_CHILD_ENV_MARKER], "1")
        self.assertEqual(observed["PATH"], "/usr/bin")
        self.assertEqual(observed["SystemRoot"], "synthetic-windows-root")
        self.assertEqual(observed["APPDATA"], "synthetic-user-appdata")
        return observed

    def test_explicit_own_profile_preserves_process_bindings(self):
        child = self._captured_child_env(profile="source")
        self.assertEqual(child["VENDOR_API_KEY"], "synthetic-source")
        self.assertEqual(child["VENDOR_PROJECT"], "source-project")

    def test_cross_profile_drops_custom_bindings_and_parent_scope(self):
        child = self._captured_child_env(
            profile="default", multiplex=True, scope={"OTHER_KEY": "synthetic-bound"})
        self.assertNotIn("VENDOR_API_KEY", child)
        self.assertNotIn("VENDOR_PROJECT", child)
        self.assertNotIn("OTHER_KEY", child)

    def test_multiplexed_own_profile_uses_bound_scope(self):
        child = self._captured_child_env(
            profile="", multiplex=True, scope={"VENDOR_API_KEY": "synthetic-bound"})
        self.assertEqual(child["VENDOR_API_KEY"], "synthetic-bound")
        self.assertNotIn("VENDOR_PROJECT", child)


    def test_multiplexed_own_delivery_pins_context_home_through_cli_startup(self):
        from agent import secret_scope
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        from hermes_cli import main as cli_main
        import sys
        with tempfile.TemporaryDirectory(prefix="cron-multiplex-home-") as tmp:
            home = Path(tmp)
            root = home / ".hermes"
            secondary = root / "profiles/secondary"
            unrelated = root / "profiles/unrelated"
            secondary.mkdir(parents=True)
            unrelated.mkdir(parents=True)
            (root / "active_profile").write_text("unrelated")
            observed = {}
            def spawn(argv, **kwargs):
                # A real child starts without its parent's ContextVars.
                child_token = set_hermes_home_override(None)
                try:
                    with patch.dict(os.environ, kwargs["env"], clear=True), patch.object(sys, "argv", list(argv)):
                        cli_main._apply_profile_override()
                        observed.update(os.environ)
                finally:
                    reset_hermes_home_override(child_token)
                return subprocess.CompletedProcess(argv, 0, "", "")
            old_active = secret_scope.is_multiplex_active()
            scope_token = secret_scope.set_secret_scope({"SECONDARY_KEY": "synthetic-secondary"})
            home_token = set_hermes_home_override(secondary)
            try:
                secret_scope.set_multiplex_active(True)
                with (
                    patch.object(Path, "home", return_value=home),
                    patch.dict(os.environ, {"HOME": str(home), "HERMES_HOME": str(root), "PATH": "/usr/bin"}, clear=True),
                    patch.object(scheduler_delivery.shutil, "which", return_value="/usr/bin/hermes"),
                    patch.object(scheduler_delivery.subprocess, "run", side_effect=spawn),
                    patch.object(scheduler_delivery, "_get_bot_chat_delivery_timeout", return_value=5),
                ):
                    self.assertIsNone(scheduler_delivery._deliver_to_bot_chat(
                        {"id": "synthetic-job"}, "Synthetic notice", ""))
            finally:
                reset_hermes_home_override(home_token)
                secret_scope.reset_secret_scope(scope_token)
                secret_scope.set_multiplex_active(old_active)
            self.assertEqual(observed["HERMES_HOME"], str(secondary))
            self.assertEqual(observed["SECONDARY_KEY"], "synthetic-secondary")


if __name__ == "__main__":
    unittest.main()
