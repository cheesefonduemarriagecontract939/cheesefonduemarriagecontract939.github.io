import type { UIStrings } from "../types";

export default {
  nav: {
    home: "首页",
    posts: "文章",
    tags: "专题 / 标签",
    readingList: "阅读清单",
    about: "关于",
    archives: "归档",
    search: "搜索",
  },
  post: {
    publishedAt: "发布于",
    updatedAt: "更新于",
    sharePostIntro: "分享这篇文章：",
    sharePostOn: "分享到 {{platform}}",
    sharePostViaEmail: "通过邮件分享这篇文章",
    tagLabel: "标签",
    backToTop: "回到顶部",
    goBack: "返回",
    editPage: "编辑页面",
    previousPost: "上一篇",
    nextPost: "下一篇",
  },
  pagination: {
    prev: "上一页",
    next: "下一页",
    page: "第",
  },
  home: {
    socialLinks: "社交链接",
    featured: "精选文章",
    recentPosts: "最新文章",
    allPosts: "查看全部文章",
  },
  footer: {
    copyright: "Copyright",
    allRightsReserved: "保留所有权利。",
  },
  pages: {
    tagTitle: "标签",
    tagDesc: "带有该标签的所有文章。",

    tagsTitle: "专题 / 标签",
    tagsDesc: "按主题快速浏览你的文章索引。",

    postsTitle: "全部文章",
    postsDesc: "这里收录所有公开发布的阅读笔记与训练笔记。",

    archivesTitle: "归档",
    archivesDesc: "按时间线回看每一次记录与迭代。",

    readingListTitle: "阅读清单",
    readingListDesc: "用一页维护正在读、计划读和已经读完的材料。",

    searchTitle: "搜索",
    searchDesc: "搜索任意文章、术语或关键词。",
  },
  a11y: {
    skipToContent: "跳到正文",
    openMenu: "打开菜单",
    closeMenu: "关闭菜单",
    toggleTheme: "切换主题",
    searchPlaceholder: "搜索文章...",
    noResults: "没有找到结果",
    goToPreviousPage: "前往上一页",
    goToNextPage: "前往下一页",
  },
  notFound: {
    title: "404 未找到页面",
    message: "页面不存在",
    goHome: "返回首页",
  },
} satisfies UIStrings;
