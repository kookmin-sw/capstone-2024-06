/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["three"],
  reactStrictMode: false,
  env: {

    Localhost: "http://10.30.117.226:8080",
    OnlyiP : "10.30.117.226:8080",
  },
  images: {
    domains: ['t1.kakaocdn.net', '10.30.117.226', 'k.kakaocdn.net'], 

    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
  },
};

module.exports = nextConfig;
