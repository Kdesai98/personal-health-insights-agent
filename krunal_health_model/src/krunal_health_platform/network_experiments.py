"""Network-aware experiment scaffolding for future social features."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from .models import jsonable, now_iso


@dataclass
class SocialEdge:
    source_user_id_hash: str
    target_user_id_hash: str
    edge_type: str = "friend"


@dataclass
class ClusterAssignment:
    cluster_id: str
    user_ids: list[str]
    treatment_arm: str


@dataclass
class NetworkExperimentPlan:
    experiment_id: str
    unit_type: str
    assignments: list[ClusterAssignment]
    exposure_mapping: str
    interference_risk: str
    generated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class NetworkExperimentDesigner:
    """Builds cluster-randomized plans when friend/cohort features exist."""

    def design_cluster_randomized_plan(
        self,
        experiment_id: str,
        user_ids: list[str],
        edges: list[SocialEdge],
        treatment_arms: list[str],
    ) -> NetworkExperimentPlan:
        clusters = self._connected_components(user_ids, edges)
        assignments = []
        for idx, cluster_users in enumerate(clusters):
            digest = sha256(f"{experiment_id}:{idx}:{','.join(sorted(cluster_users))}".encode("utf-8")).hexdigest()
            arm = treatment_arms[int(digest[:8], 16) % len(treatment_arms)]
            assignments.append(
                ClusterAssignment(
                    cluster_id=f"cluster_{idx}",
                    user_ids=sorted(cluster_users),
                    treatment_arm=arm,
                )
            )
        interference_risk = "none_single_user" if len(user_ids) <= 1 else "managed_by_cluster_randomization"
        return NetworkExperimentPlan(
            experiment_id=experiment_id,
            unit_type="cluster",
            assignments=assignments,
            exposure_mapping="user exposure equals assigned cluster arm; neighbor exposure is tracked separately",
            interference_risk=interference_risk,
        )

    def _connected_components(self, user_ids: list[str], edges: list[SocialEdge]) -> list[list[str]]:
        graph = {user_id: set() for user_id in user_ids}
        for edge in edges:
            graph.setdefault(edge.source_user_id_hash, set()).add(edge.target_user_id_hash)
            graph.setdefault(edge.target_user_id_hash, set()).add(edge.source_user_id_hash)

        seen = set()
        components = []
        for user_id in sorted(graph):
            if user_id in seen:
                continue
            stack = [user_id]
            component = []
            seen.add(user_id)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            components.append(component)
        return components
