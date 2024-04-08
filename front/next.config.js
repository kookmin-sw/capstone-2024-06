/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  env: {
    Localhost: "http://10.223.118.159:8080",
    OnlyiP: "10.223.118.159:8080",
  },
  images: {
    domains: ['t1.kakaocdn.net', '10.223.118.159', 'k.kakaocdn.net'],
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
  },
};

module.exports = nextConfig;
