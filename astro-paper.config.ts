import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://bakuma-sea.github.io/",
    title: "Bakuma Notes",
    description: "记录 RL、LLM 训练与阅读笔记的中文博客。",
    author: "Bakuma",
    profile: "https://github.com/Bakuma-sea",
    ogImage: "default-og.jpg",
    lang: "zh-CN",
    timezone: "Asia/Shanghai",
    dir: "ltr",
  },
  posts: {
    perPage: 6,
    perIndex: 5,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: false,
    showArchives: true,
    showBackButton: true,
    editPost: { enabled: false },
    search: "pagefind",
  },
  socials: [
    {
      name: "github",
      url: "https://github.com/Bakuma-sea",
      linkTitle: "Bakuma 的 GitHub",
    },
  ],
  shareLinks: [],
});
