# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities privately through a
[GitHub security advisory](https://github.com/arnav-goel10/whisper-singapore-english/security/advisories/new)
rather than a public issue. Include reproduction steps and the affected commit.
Expect an acknowledgement within seven days.

## Scope

This is a research repository. It makes no network calls of its own, reads no
credentials, and ships no model weights or corpus data.

Two classes of issue matter here. Supply chain: vulnerabilities in the optional
`asr` dependencies, which pull model code and audio decoders. And data
disclosure: any change that would cause licensed corpus content, per-utterance
predictions, or reference transcripts to be written into the repository. Loading
a model from an untrusted `--adapter` path executes third-party code, so treat
adapter paths as you would any downloaded artifact.
