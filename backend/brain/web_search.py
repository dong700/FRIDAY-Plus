"""
联网搜索模块 - Tavily API 集成

功能：
1. 调用 Tavily Search API 进行网络搜索
2. 大脑自主决定是否需要搜索（LLM 判断）
3. 搜索结果注入到对话上下文中
"""

import aiohttp
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("brain.web_search")


@dataclass
class SearchResult:
    """单条搜索结果"""
    title: str
    url: str
    content: str
    score: float


@dataclass
class SearchResponse:
    """搜索响应"""
    query: str
    results: List[SearchResult]
    answer: str  # Tavily AI 生成的摘要回答
    success: bool
    error: str = ""


class TavilyClient:
    """Tavily 搜索客户端"""

    def __init__(self, api_key: str, max_results: int = 5):
        self.api_key = api_key
        self.max_results = max_results
        self.base_url = "https://api.tavily.com"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15.0)
            )
        return self._session

    async def search(self, query: str, search_depth: str = "basic",
                     include_answer: bool = True,
                     max_results: int = None) -> SearchResponse:
        """
        执行搜索

        Args:
            query: 搜索关键词
            search_depth: "basic" | "advanced"
            include_answer: 是否让 Tavily AI 生成摘要
            max_results: 最大结果数
        """
        if not query.strip():
            return SearchResponse(query="", results=[], answer="", success=False, error="空查询")

        session = await self._get_session()
        url = f"{self.base_url}/search"

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": False,
            "max_results": max_results or self.max_results,
        }

        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()

                    results = []
                    for item in data.get("results", []):
                        results.append(SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            content=item.get("content", ""),
                            score=item.get("score", 0),
                        ))

                    return SearchResponse(
                        query=query,
                        results=results,
                        answer=data.get("answer", ""),
                        success=True,
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Tavily search error: {response.status} - {error_text}")
                    return SearchResponse(query=query, results=[], answer="",
                                         success=False, error=f"HTTP {response.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Tavily network error: {e}")
            return SearchResponse(query=query, results=[], answer="",
                                 success=False, error=f"网络错误: {str(e)}")
        except Exception as e:
            logger.error(f"Tavily search exception: {e}")
            return SearchResponse(query=query, results=[], answer="",
                                 success=False, error=str(e))

    async def extract(self, urls: List[str]) -> Dict[str, Any]:
        """提取指定 URL 的内容"""
        if not urls:
            return {"success": False, "error": "无URL"}

        session = await self._get_session()
        url = f"{self.base_url}/extract"

        payload = {
            "api_key": self.api_key,
            "urls": urls[:5],  # 最多5个
        }

        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return {"success": True, "data": await response.json()}
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Tavily extract error: {e}")
            return {"success": False, "error": str(e)}

    def format_results_for_context(self, response: SearchResponse) -> str:
        """将搜索结果格式化为可注入对话上下文的文本"""
        if not response.success or not response.results:
            return ""

        lines = [f"\n[联网搜索] \"{response.query}\" 的搜索结果："]

        if response.answer:
            lines.append(f"\n摘要: {response.answer}")

        for i, r in enumerate(response.results[:3], 1):  # 最多3条
            lines.append(f"\n{i}. {r.title}")
            lines.append(f"   {r.content[:300]}")
            lines.append(f"   来源: {r.url}")

        return "\n".join(lines)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# 单例
_tavily_client: Optional[TavilyClient] = None


def get_tavily_client(api_key: str = None, **kwargs) -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        if not api_key:
            from core.config import get_config
            config = get_config()
            api_key = getattr(config.tavily, 'api_key', '') if hasattr(config, 'tavily') else ''
        _tavily_client = TavilyClient(api_key=api_key, **kwargs)
    return _tavily_client
