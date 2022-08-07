# How to Write Better Git Commit Messages

https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/

1. 첫 글자는 대문자로. 문장기호로 끝내지 말 것
2. 제목은 명령형으로 작성할 것. `Added ~` 가 아니라 `Add ~ ` 형태로
3. commit의 타입을 명시하기
4. 첫 줄은 50자 이내, body는 72자 이내
5. `maybe` `I think` 같은 거 쓰지 말기

- 왜 이렇게 수정했는가?
- 이 변경사항이 어떤 영향을 끼칠까?
- 왜 이런 수정이 필요했는가?
- 어느 부분이 변경되었는가?

## Conventional Commits

```
fix: fix foo to enable bar

This fixes the broken behavior of the component by doing xyz. 

BREAKING CHANGE
Before this fix foo wasn't enabled at all, behavior changes from <old> to <new>

Closes D2IQ-12345
```

첫 줄은 subject line이다. type과 제목 작성

- feat: 새로운 기능
- fix: 버그 수정
- chore: src나 test 파일 건드리지 않는 잡일
- refactor: 리팩터링
- docs: 문서
- style: 스타일 포맷팅
- test: 테스트 관련
- perf: 성능 향상
- ci: 유지보수 관련
- build: 빌드 관련
- revert: 이전 커밋을 되돌림

다음 줄에서는 설명을 작성한다.

footer의 기호는 JIRA 같은 협업 툴 관련한 내용