# MLOps Hardening Design

## Goal

ทำให้ training, model artifact, promotion และ serving ทำงานสอดคล้องกันมากขึ้น และให้ validation/error message ปลอดภัยและแก้ปัญหาได้จริง

## Scope

- ตรวจสอบ prediction horizon และ input consistency ก่อนเรียก AutoGluon
- เพิ่ม prediction metadata ใน model manifest
- ทำให้ quality gate ใช้ threshold จาก environment ครบ
- ทำให้ artifact cache ตรวจสอบ URI/checksum ก่อน reuse
- ลดความเสี่ยงจาก fallback ของ feature artifact
- ทำให้ promotion ตรวจสอบ checksum และแยก pointer update จาก serving rollout อย่างชัดเจน
- harden Docker Compose defaults สำหรับ local use
- เพิ่ม unit/integration-style tests และเอกสารคำสั่งใช้งาน

## Non-goals

- ยังไม่เพิ่มระบบ Kubernetes หรือ external monitoring backend
- ยังไม่เปลี่ยน model algorithm หรือ feature definitions
- ยังไม่ทำ authentication แบบ production-grade ในรอบเดียวกัน แต่จะไม่เปิด default deployment ให้เข้าใจผิดว่าเป็น production

## Acceptance criteria

1. Invalid horizon/item/date requests return HTTP 422 with actionable details.
2. A newly packaged model manifest contains prediction length and archive checksum.
3. Quality gate receives RMSLE, WQL, and RMSE thresholds from environment.
4. Reusing a cache directory with a different model URI cannot silently serve the old model.
5. Promotion validates artifact existence and checksum metadata before writing the pointer.
6. Root and serving tests, lint, compose validation, and serving image build checks pass.
7. A fresh training run produces an artifact that loads and serves successfully.
