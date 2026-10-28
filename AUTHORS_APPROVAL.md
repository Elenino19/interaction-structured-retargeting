# Author and citation approval

Draft v6.2 has one paper author: **Buwen Zhang**. There are no other paper authors. Affiliation, email, ORCID, venue, DOI, and publication status were not supplied and are not guessed.

The checked PDF file still prints an anonymous byline. That file is outside this repository and was not edited. This record is the non-anonymous author list for the release.

## Roles

| Role | Who | Status |
|---|---|---|
| Paper author | Buwen Zhang, sole author | Confirmed for Draft v6.2 |
| Original software developer | Buwen Zhang, sole developer of the original project code | Confirmed. Third-party code, data, models, and assets stay under their own licenses |
| Third-party contributors | CORE4D, SMPL-X / Max-Planck, Unitree, GVHMR / ZJU 3DV, and dependency authors | `THIRD_PARTY_LICENSES.md`. Not software authors |

Being the paper author does not by itself license the code. Third-party datasets stay out of both author lists.

## Still pending

- A project `LICENSE`. MIT is still only a candidate. Do not add it until it is explicitly approved. Third-party materials are not covered by that candidate.
- Affiliation, email, and ORCID, only if the author later supplies and approves them.
- Repository URL to replace the omitted code URL.
- After publication only: year, venue or journal, DOI, and paper URL.

A minor revision of Draft v6.2 does not add those facts. Follow `PAPER_VERSION_SYNC.md` when the manuscript text changes.

## Double-blind submission

No venue has been named. If the submission venue requires double-blind review, the submitted manuscript and any public page must omit the author name. Keep using the anonymized PDF for that submission. This private release tree may keep the name. The local page under `docs/` names the author and is not published.

## Software citation

`CITATION.cff` is software-first (`type: software`). It lists one author:

```yaml
authors:
  - given-names: Buwen
    family-names: Zhang
```

No affiliation and no ORCID are included. Do not add `license` unless `RELEASE_LICENSE_AUDIT.md` says a project license was approved.

## Paper citation after publication

Do not add `preferred-citation` until a bibliographic record exists. When it does, the paper author block stays a single person:

```yaml
preferred-citation:
  type: conference-paper
  title: "Interaction-Structured Motion Retargeting for Multi-Party Humanoid–Object Interactions"
  authors:
    - given-names: Buwen
      family-names: Zhang
  year: YYYY
  conference:
    name: "..."
  doi: "..."
```

Use `type: article` and `journal` if the published work is a journal article. Fill year, venue, DOI, and URL only from the published record. If the title changes before publication, change this title and the software `title` together.
