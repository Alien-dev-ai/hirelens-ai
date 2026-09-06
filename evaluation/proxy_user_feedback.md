# Proxy User Evaluation

## 1. Purpose

This evaluation was conducted to get an initial usability and
output-comprehension signal from someone outside the builder — i.e.,
whether a person who did not write HireLens could open its output and
understand what it was telling them, without any walkthrough from the
developer.

**This was not validation by a professional recruiter and should not be
interpreted as evidence of recruiter adoption.** It is a single, informal
session with one proxy user, recorded honestly as exactly that — not as a
substitute for the recruiter/HR-professional testing the project still
needs (see §7 and §9).

## 2. Proxy User Context

- The proxy user was not the developer.
- The proxy user was not an HR professional or recruiter.
- The proxy user was familiar enough with job descriptions, CV
  terminology, and candidate requirements to understand the system's
  output.
- Therefore, this session is treated as an **informal proxy-user usability
  evaluation**, not domain-expert validation.

No further demographic, job-title, or background detail is recorded here,
because none beyond the above was established for this session.

## 3. Evaluation Setup

Two real candidate documents were run against the same job description, for
a **Video Editor** role, so both runs could be compared against identical
requirements.

**Video Editor job description — requirements used:**

**Required:**
- Proven experience as a Video Editor or similar role
- Adobe Premiere Pro
- Adobe After Effects
- Storytelling, pacing, and visual composition
- Social media editing
- Video formats and resolutions
- Color correction and grading
- Audio editing and synchronization
- Attention to detail
- Managing multiple projects and deadlines

**Preferred:**
- DaVinci Resolve
- Motion graphics
- YouTube editing
- TikTok and Instagram Reels editing
- Adobe Photoshop
- Cameras and basic cinematography
- Relevant Film Production, Media, or Communication education

### Scenario A — More relevant candidate profile

The candidate profile contained documented skills and experience related
to:

- Video editing
- Motion graphics
- Color grading
- Cinematic editing
- Short-form content editing
- Storytelling
- 3D animation
- Social media content

The system produced:

- **Overall Match Score: 44/100**
- **Alignment label: Limited Alignment**

The output found documented evidence for several relevant capabilities,
including storytelling, social-media content, color grading, and motion
graphics. The system also correctly identified many requirements that
needed verification or had no documented evidence, including specific
software tools and work-history evidence.

### Scenario B — Deliberately mismatched candidate profile

A second candidate CV, with a software/backend-oriented profile, was
tested against the same Video Editor job description as a contrast case.
The candidate's documented skills included areas such as:

- Core Java
- SQL
- APIs
- Spring Framework
- Hibernate
- Maven
- Spring Boot
- JVM
- JDBC

The system produced:

- **Overall Match Score: 4/100**
- **Alignment label: Minimal Alignment**

The system found almost no documented evidence supporting the Video Editor
requirements. The only partial/ambiguous evidence identified was related
to attention to detail, surfaced through the candidate's technical
documentation and data-integrity work.

This second scenario was used as a contrast test, to observe whether the
system would clearly distinguish a strongly mismatched profile from the
more relevant candidate profile in Scenario A. **Two examples are not a
statistically meaningful sample** — no claim of statistical accuracy is
made from this contrast alone.

## 4. Observed System Behavior

- The more relevant profile (Scenario A) received **44/100** and
  **Limited Alignment**.
- The strongly mismatched software/backend profile (Scenario B) received
  **4/100** and **Minimal Alignment**.
- The output clearly separated the three evidence states: **Evidence
  Found**, **Needs Verification**, and **No Evidence Found**.
- The system maintained the principle that the score is decision support,
  not a hiring decision — no hire/no-hire language appeared in either
  run's output.
- The system generated a recruiter analysis, a requirement-by-requirement
  evidence breakdown, key strengths, key gaps, and interview validation
  areas for both scenarios.

One analysis run was observed to take approximately **4 minutes and 19
seconds**. This is a single observed run's latency, not an average or a
benchmark across multiple runs — it is reported here only because it was
directly observed during this session, not because it is claimed to be
representative.

## 5. Proxy User Feedback

### Clarity

The proxy user reported that it was clear what the system had reviewed
and what information the output was presenting.

### Most useful output

The proxy user found these parts particularly useful:

1. The requirement breakdown / requirement-level results
2. The Recruiter Analysis

These sections made it easier to understand what the candidate had, and
what was missing or required further verification, without having to
re-derive that from the raw CV and job description.

### Evidence status model

The proxy user said the distinction between **Evidence Found**, **Needs
Verification**, and **No Evidence Found** made sense.

However, the proxy user suggested that the wording **"No Evidence Found"**
may be clearer than terminology such as "Not Evidenced" or
"Non-evidenced." This is recorded as a wording preference from this
session; it has not been confirmed elsewhere in the repository whether
this change has already been made, and it should not be assumed
implemented without checking the current output copy directly.

### Confusion

The proxy user reported that nothing in the reviewed output was
significantly confusing.

### Perceived usefulness

The proxy user said the system would be useful when reviewing candidates
or recruiting, because it helps organize the candidate's documented
strengths, gaps, and requirements that need verification.

## 6. Improvement Suggested by the Proxy User

The proxy user suggested that the system could provide a faster
high-level summary of the candidate's overall alignment — something that
could quickly communicate the AI's evidence-based assessment, for
example:

- The candidate has strengths in specific areas.
- Important requirements are missing or unverified.
- Based on documented evidence, the profile currently shows limited
  alignment with the role.

This is **not** a suggestion for an automatic hiring recommendation, and
is not documented as one. The suggestion creates a genuine design tension,
because HireLens deliberately avoids making hiring decisions of any kind
— a fast "verdict-style" summary risks reading as exactly that if it is
not carefully worded.

A possible future improvement, consistent with HireLens' existing
guardrails, could be a clearly labeled **"Evidence-Based Alignment
Summary"** — rather than language like "Reject this candidate," "Not
suitable," or "Do not hire." Such a summary would communicate the overall
documented alignment while preserving the existing rule that the
recruiter makes the final hiring decision.

This feedback is recorded here as a **future UX consideration**. It is
not necessarily implemented in the current version, and no code has been
changed as a result of it.

## 7. What This Evaluation Does and Does Not Demonstrate

### What it demonstrates

- A person other than the builder interacted with and reviewed the
  system's output.
- The proxy user understood the major evidence-status categories.
- The proxy user identified the requirement-level breakdown and recruiter
  analysis as particularly useful.
- The system produced substantially different alignment scores for a more
  relevant profile (44/100) and a deliberately mismatched profile
  (4/100).
- The evaluation generated a concrete UX improvement suggestion.

### What it does not demonstrate

- Recruiter validation
- HR domain-expert validation
- Real-world adoption
- Measured time savings
- Hiring accuracy
- Predictive validity
- Statistical performance
- A representative usability study

This was **one informal proxy-user evaluation** and should be treated as
an early qualitative signal — not as evidence that stands in for
recruiter or HR-professional testing.

## 8. Key Takeaways

1. The proxy user found the system understandable.
2. Requirement-level evidence and recruiter analysis were the most useful
   areas of the output.
3. The evidence-state terminology was understandable, with a wording
   preference for "No Evidence Found."
4. The proxy user suggested a faster, high-level, evidence-based
   alignment summary.
5. The contrast test produced a 44/100 score for the more relevant
   profile and a 4/100 score for the strongly mismatched profile.
6. One observed analysis run took 4 minutes 19 seconds, which highlights
   latency as an important area for improvement.
7. Further testing with actual recruiters or HR professionals remains
   necessary.

## 9. Next Steps

1. Test with at least one recruiter or HR professional.
2. Collect feedback from more than one proxy or target user.
3. Measure usability and time-per-review more systematically.
4. Investigate ways to reduce analysis latency.
5. Consider a clearly labeled evidence-based alignment summary that gives
   a faster overview without making a hiring decision.
6. Test additional deliberately mismatched and partially matched
   candidate profiles.
