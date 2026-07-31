# Package initializer for the VocosFormer experiment, a
# literature-derived, parameter-matched Vocos adaptation reimplementing
# the WavTokenizer attention position network with no novelty claim in
# the package name; modules are imported by full path.
#
# Design decisions:
# - The export list is deliberately empty rather than re-exporting the
#   module and its configuration, so nothing can reach this row except by
#   naming the full module path or by going through the registry, and the
#   package name alone never reads as an endorsement of the adaptation
#
# Author: Rahul Sawhney

__all__: list[str] = []
