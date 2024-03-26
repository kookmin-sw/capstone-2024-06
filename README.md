[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=A16D07&random=false&width=435&lines=%EC%9D%B4+%EC%B1%85%EC%83%81+%EC%96%B4%EB%96%A4%EB%8D%B0%3F+++%EC%96%B4%EB%96%A4%EB%8D%B0%EC%8A%A4%ED%81%AC(what_desk))](https://git.io/typing-svg)

> 2024 KMU SW 캡스톤 디자인 06조
> https://kookmin-sw.github.io/capstone-2024-06/

---

## 📔 목차

- [ 프로젝트 소개](#-프로젝트-소개)
- [ 기술스택](#-기술-스택)
- [ 팀 소개](#-팀-소개)
- [ 주요 기능](#-주요-기능)
- [ 사용법](#-사용법)
- [ 시연 영상](#-시연-영상)

---

### 1. 프로젝트 소개
"어떤데스크 (what_desk)"는 현대 사회에서 가장 많이 사용되는 책상을 중심으로 한 일의 효율성과 생산성을 증진시키기 위한 인테리어 디자인에 관한 것입니다. 이 프로젝트는 사용자에게 어울리는 책상 디자인을 추천하는 것뿐만 아니라, 사용자의 책상과 여러 사람들의 책상을 분석하여 다양한 정보를 제공하고 있습니다.

어떤데스크는 사용자의 책상을 분석하여 그에 맞는 적절한 디자인을 추천하는데 그치지 않고, 사용자들에게 다양한 정보를 제공하려고 합니다. 데스크테리어에 관련하여 어떤 키워드가 많이 검색되는지 혹은 어떤 아이템들이 인기가 많은지 알려주고 사용자의 책상 정보를 분석하여 시각적으로 제공해 사용자에게 데스크테리어의 가이드라인이 되어 줄 것입니다.

어떤데스크는 커뮤니티 기능을 통해 사용자들끼리 정보를 공유하고 소통할 수 있는 플랫폼을 제공하고 있습니다. 사용자들은 자신의 작업 공간을 공유하고 다른 사용자들의 경험을 듣고 의견을 나눌 수 있습니다. 이를 통해 사용자들은 더 나은 작업 환경을 조성하는 데 도움을 받을 수 있습니다. 뿐만 아니라 사용자들간의 활발한 물건 거래도 지향하고 있습니다.
 
어떤데스크는 데스크테리어를 쉽게 만들어 사용자들이 작업 문화를 개선하고 생산성을 향상시키고 사용자들이 자신의 개성을 표현하는데 도움을 주려고 합니다.

---
### Abstract
"What_desk" is about interior design focused on the most commonly used desks in modern society to enhance efficiency and productivity in work. This project not only recommends suitable desk designs to users but also provides various information by analyzing users' desks and those of others.

Beyond recommending appropriate designs based on users' desks, What_desk aims to provide users with diverse information. It informs users about popular keywords related to desk decor and trending items, visually analyzing users' desk information to offer guidelines for desk decoration.

Through its community feature, What_desk provides a platform for users to share information and communicate. Users can share their workspaces, listen to others' experiences, and exchange opinions, thereby fostering a better working environment. Additionally, What_desk encourages active trading among users.

What_desk aims to facilitate the creation of desk decor, helping users improve their work culture, enhance productivity, and express their individuality.

---

### 2. 기술 스택

<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 

### 3. 팀 소개

| 이름 | 사진 | 학번 | 역할 | 깃허브 주소 |
|---|---|---|---|---|
| 박승현 | ![박승현 이미지](back/readme_image/깃허브 프사.jpeg) | xxxx1595 | Team Leader, BackEnd, AI | [박승현 GitHub](https://github.com/gustmdqkr321) |
| 마준영 | ![마준영 이미지](마준영_이미지_URL) | xxxx | BackEnd, AI | [마준영 GitHub](마준영_GitHub_URL) |
| 박근우 | ![박근우 이미지](박근우_이미지_URL) | xxxx | FrontEnd, Design | [박근우 GitHub](박근우_GitHub_URL) |
| 윤유진 | ![윤유진 이미지](윤유진_이미지_URL) | xxxx | FrontEnd, Design | [윤유진 GitHub](윤유진_GitHub_URL) |
| 조한울 | ![조한울 이미지](조한울_이미지_URL) | xxxx | FrontEnd, Design | [조한울 GitHub](조한울_GitHub_URL) |

---

### 4. 주요 기능

  1. **책상 사진 추천 및 디자인 기능:**
      - 사용자가 책상 사진을 업로드하면 해당 책상에 어울리는 디자인(스탠드, 의자 등)을 추천
      - 사용자가 설정한 아이템을 기준으로 추천을 제공할수도 O
  2. **사용자 선호도 및 추천 순위:**
      - 사용자가 선택한 아이템이나 디자인에 대한 선호도를 기록하여 추천에 반영
      - 다른 사용자들이 많이 선택한 아이템이나 디자인을 순위로 공유가능
  3. **상품 추천 및 구매처 링크:**
      - 추천된 제품의 가격 및 사용자 사진에 자주 등장하는 상품을 추천
      - 구매처 링크를 제공하여 사용자가 실제로 구매할 수 있도록 연결
  4. **인테리어 키워드 추천:**
      - 인테리어에 익숙하지 않은 초보 사용자가 처음부터 인테리어를 시작할 때 키워드를 추천
      - 구글 검색 결과나 다른 플랫폼의 트렌드를 분석하여 키워드를 추출
  5. **사진에서 추출된 물건 가격 및 크롤링:**
      - 추천된 물건의 가격을 제공하고, 크롤링을 통해 실제 구매 가능한 상품 정보를 제공
  6. **커뮤니티 기능:**
      - 로그인 및 회원가입을 통해 커뮤니티 이용이 가능하게 함
      - 중고 거래 기능을 추가하여 사용자간 거래가 가능토록함
      - 사용자가 원하는 아이템을 체크박스로 선택가능토록
  7. **투표 및 사용자 참여:**
      - 만든 사진 중에서 투표를 통해 인기 있는 사진을 투표가능토록
      - 커뮤니티에서  사용자 간 토론도 가능토록
  8. **커뮤니티에 추가될 기능:**
      - 로그인, 회원가입, 게시판, 검색어 인기 순위, 중고 거래 게시판, 인공지능을 활용한 특별한 기능 등을 포함

### 5. 사용법

소스코드제출시 설치법이나 사용법을 작성하세요.

### 6. 시연 영상