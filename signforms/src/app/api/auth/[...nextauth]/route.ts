//Users/user/Desktop/signforms/signforms/src/app/api/auth/[...nextauth]/route.ts

import NextAuth from "next-auth";
import { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import GoogleProvider from "next-auth/providers/google";
import KakaoProvider from "next-auth/providers/kakao";
import NaverProvider from "next-auth/providers/naver";

//자체 로그인

const credentialsProvider = CredentialsProvider({
  // The name to display on the sign in form (e.g. "Sign in with...")
  name: 'credentials',
  // `credentials` is used to generate a form on the sign in page.
  // You can specify which fields should be submitted, by adding keys to the `credentials` object.
  // e.g. domain, username, password, 2FA token, etc.
  // You can pass any HTML attribute to the <input> tag through the object.
  credentials: {
    id: {
      label: 'username',
      type: 'text',
      placeholder: '아이디를 입력하세요',
    },
    password: {
      label: 'password',
      type: 'password',
      placeholder: '비밀번호를 입력하세요',
    },
  },

  async authorize(credentials, req) {
    try {
      const res = await fetch("http://210.178.142.51:8001/user/sign_in", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          id: credentials?.id,
          password: credentials?.password,
        }),
      });

      if (res.ok) {
        const user = await res.json();
        console.log(user);

        if (user) {
          // 사용자 정보 반환
          return user;
        } else {
          // 사용자 정보가 없는 경우
          console.log('Incorrect password');
          alert('Incorrect password');
        }
      } else {
        console.error("POST 요청이 실패했습니다.");
        // 요청이 실패한 경우 에러 처리
        return null;
      }
    } catch (error) {
      console.error("POST 요청 중 에러가 발생했습니다.", error);
      // 요청 중 에러 발생 시 처리
      return null;
    }
  }
});

// 소셜로그인
// 카카오 로그인 버튼
const kakaoCustomProvider = KakaoProvider({
  clientId: process.env.KAKAO_CLIENT_ID || '',
  clientSecret: process.env.KAKAO_CLIENT_SECRET || '',
});

kakaoCustomProvider.style = {
  logo: 'https://developers.kakao.com/tool/resource/static/img/button/kakaotalksharing/kakaotalk_sharing_btn_small.png',
  logoDark: 'https://developers.kakao.com/tool/resource/static/img/button/kakaotalksharing/kakaotalk_sharing_btn_small.png',
  bgDark: '#FEE500',
  bg: '#FEE500',
  text: '#191919',
  textDark: '#191919',
};

// 네이버 로그인 버튼
const naverCustomProvider = NaverProvider({
  clientId: process.env.NAVER_CLIENT_ID || '',
  clientSecret: process.env.NAVER_CLIENT_SECRET || '',
});

naverCustomProvider.style = {
  logo: 'https://logoproject.naver.com/favicon.ico',
  logoDark: 'https://logoproject.naver.com/favicon.ico',
  bgDark: '#2DB400',
  bg: '#2DB400',
  text: '#FFFFFF',
  textDark: '#FFFFFF',
};

const authOptions: NextAuthOptions = {
  // 사용할 인증 프로바이더들을 설정
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID || "",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || "",
    }),
    kakaoCustomProvider,
    naverCustomProvider,
    credentialsProvider,
  ],
  // 세션의 비밀 값으로 사용할 문자열
  secret: process.env.NEXTAUTH_SECRET,
  // 페이지 경로 설정
  pages: {
    // 오류 발생 시 이동할 페이지 경로 설정
    error: '/error',
    // 로그아웃 시 이동할 페이지 경로 설정
    signOut: '/'
  },
  // 세션 설정
  session: {
    // 세션 저장 방식 설정
    strategy: 'jwt',
    // 세션의 최대 수명 설정 (초 단위)
    maxAge: 24 * 60 * 60,
    // 세션 업데이트 주기 설정 (초 단위)
    updateAge: 2 * 24 * 60 * 60
  },
  // 콜백 함수 설정
  callbacks: {
    // JWT 토큰 처리 콜백 함수
    async jwt({ token, account }) {
      if (account) {
        token.accessToken = account.access_token;
      }
      return token;
    },
    // 세션 처리 콜백 함수
    async session({ session, token, user }) {
      return session;
    },
    // 리다이렉트 처리 콜백 함수
    async redirect({ url, baseUrl }) {
      // 홈으로
      return `${baseUrl}/`;
    },
  },
};

// NextAuth 초기화
const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
