/* prettier-ignore-start */

/* eslint-disable */

// @ts-nocheck

// noinspection JSUnusedGlobalSymbols

// This file is auto-generated by TanStack Router

// Import Routes

import { Route as rootRoute } from './routes/__root'
import { Route as AuthImport } from './routes/_auth'
import { Route as IndexImport } from './routes/index'
import { Route as AuthSignupImport } from './routes/_auth/signup'
import { Route as AuthSigninImport } from './routes/_auth/signin'
import { Route as AuthConfirmImport } from './routes/_auth/confirm'
import { Route as AuthProtectedImport } from './routes/_auth/_protected'
import { Route as AuthProtectedLayoutImport } from './routes/_auth/_protected/_layout'
import { Route as AuthProtectedLayoutHomeImport } from './routes/_auth/_protected/_layout/home'
import { Route as AuthProtectedLayoutDebugImport } from './routes/_auth/_protected/_layout/debug'

// Create/Update Routes

const AuthRoute = AuthImport.update({
  id: '/_auth',
  getParentRoute: () => rootRoute,
} as any)

const IndexRoute = IndexImport.update({
  path: '/',
  getParentRoute: () => rootRoute,
} as any)

const AuthSignupRoute = AuthSignupImport.update({
  path: '/signup',
  getParentRoute: () => AuthRoute,
} as any)

const AuthSigninRoute = AuthSigninImport.update({
  path: '/signin',
  getParentRoute: () => AuthRoute,
} as any)

const AuthConfirmRoute = AuthConfirmImport.update({
  path: '/confirm',
  getParentRoute: () => AuthRoute,
} as any)

const AuthProtectedRoute = AuthProtectedImport.update({
  id: '/_protected',
  getParentRoute: () => AuthRoute,
} as any)

const AuthProtectedLayoutRoute = AuthProtectedLayoutImport.update({
  id: '/_layout',
  getParentRoute: () => AuthProtectedRoute,
} as any)

const AuthProtectedLayoutHomeRoute = AuthProtectedLayoutHomeImport.update({
  path: '/home',
  getParentRoute: () => AuthProtectedLayoutRoute,
} as any)

const AuthProtectedLayoutDebugRoute = AuthProtectedLayoutDebugImport.update({
  path: '/debug',
  getParentRoute: () => AuthProtectedLayoutRoute,
} as any)

// Populate the FileRoutesByPath interface

declare module '@tanstack/react-router' {
  interface FileRoutesByPath {
    '/': {
      id: '/'
      path: '/'
      fullPath: '/'
      preLoaderRoute: typeof IndexImport
      parentRoute: typeof rootRoute
    }
    '/_auth': {
      id: '/_auth'
      path: ''
      fullPath: ''
      preLoaderRoute: typeof AuthImport
      parentRoute: typeof rootRoute
    }
    '/_auth/_protected': {
      id: '/_auth/_protected'
      path: ''
      fullPath: ''
      preLoaderRoute: typeof AuthProtectedImport
      parentRoute: typeof AuthImport
    }
    '/_auth/confirm': {
      id: '/_auth/confirm'
      path: '/confirm'
      fullPath: '/confirm'
      preLoaderRoute: typeof AuthConfirmImport
      parentRoute: typeof AuthImport
    }
    '/_auth/signin': {
      id: '/_auth/signin'
      path: '/signin'
      fullPath: '/signin'
      preLoaderRoute: typeof AuthSigninImport
      parentRoute: typeof AuthImport
    }
    '/_auth/signup': {
      id: '/_auth/signup'
      path: '/signup'
      fullPath: '/signup'
      preLoaderRoute: typeof AuthSignupImport
      parentRoute: typeof AuthImport
    }
    '/_auth/_protected/_layout': {
      id: '/_auth/_protected/_layout'
      path: ''
      fullPath: ''
      preLoaderRoute: typeof AuthProtectedLayoutImport
      parentRoute: typeof AuthProtectedImport
    }
    '/_auth/_protected/_layout/debug': {
      id: '/_auth/_protected/_layout/debug'
      path: '/debug'
      fullPath: '/debug'
      preLoaderRoute: typeof AuthProtectedLayoutDebugImport
      parentRoute: typeof AuthProtectedLayoutImport
    }
    '/_auth/_protected/_layout/home': {
      id: '/_auth/_protected/_layout/home'
      path: '/home'
      fullPath: '/home'
      preLoaderRoute: typeof AuthProtectedLayoutHomeImport
      parentRoute: typeof AuthProtectedLayoutImport
    }
  }
}

// Create and export the route tree

export const routeTree = rootRoute.addChildren({
  IndexRoute,
  AuthRoute: AuthRoute.addChildren({
    AuthProtectedRoute: AuthProtectedRoute.addChildren({
      AuthProtectedLayoutRoute: AuthProtectedLayoutRoute.addChildren({
        AuthProtectedLayoutDebugRoute,
        AuthProtectedLayoutHomeRoute,
      }),
    }),
    AuthConfirmRoute,
    AuthSigninRoute,
    AuthSignupRoute,
  }),
})

/* prettier-ignore-end */

/* ROUTE_MANIFEST_START
{
  "routes": {
    "__root__": {
      "filePath": "__root.tsx",
      "children": [
        "/",
        "/_auth"
      ]
    },
    "/": {
      "filePath": "index.tsx"
    },
    "/_auth": {
      "filePath": "_auth.tsx",
      "children": [
        "/_auth/_protected",
        "/_auth/confirm",
        "/_auth/signin",
        "/_auth/signup"
      ]
    },
    "/_auth/_protected": {
      "filePath": "_auth/_protected.tsx",
      "parent": "/_auth",
      "children": [
        "/_auth/_protected/_layout"
      ]
    },
    "/_auth/confirm": {
      "filePath": "_auth/confirm.tsx",
      "parent": "/_auth"
    },
    "/_auth/signin": {
      "filePath": "_auth/signin.tsx",
      "parent": "/_auth"
    },
    "/_auth/signup": {
      "filePath": "_auth/signup.tsx",
      "parent": "/_auth"
    },
    "/_auth/_protected/_layout": {
      "filePath": "_auth/_protected/_layout.tsx",
      "parent": "/_auth/_protected",
      "children": [
        "/_auth/_protected/_layout/debug",
        "/_auth/_protected/_layout/home"
      ]
    },
    "/_auth/_protected/_layout/debug": {
      "filePath": "_auth/_protected/_layout/debug.tsx",
      "parent": "/_auth/_protected/_layout"
    },
    "/_auth/_protected/_layout/home": {
      "filePath": "_auth/_protected/_layout/home.tsx",
      "parent": "/_auth/_protected/_layout"
    }
  }
}
ROUTE_MANIFEST_END */
