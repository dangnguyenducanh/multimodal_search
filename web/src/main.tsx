import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { ComparePage, HomePage, NotFoundPage, ProductPage, SearchPage } from "@/pages";
import "@/styles.css";

const router = createBrowserRouter([
  { path: "/", element: <HomePage /> },
  { path: "/search", element: <SearchPage /> },
  { path: "/compare", element: <ComparePage /> },
  { path: "/product/:id", element: <ProductPage /> },
  { path: "*", element: <NotFoundPage /> },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
